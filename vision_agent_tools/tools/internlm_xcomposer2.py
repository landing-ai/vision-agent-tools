import torch
from PIL import Image
from vision_agent_tools.types import VideoNumpy
from vision_agent_tools.tools.shared_types import BaseTool

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class InternLMXComposer2(BaseTool):
    _HF_MODEL = "internlm/internlm-xcomposer2d5-7b"
    MAX_NUMBER_OF_FRAMES = 32
    MAX_IMAGE_SIZE = 1024

    def _transform_image(image: Image.Image, max_size: int):
        image = image.convert("RGB")
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size))
        return image

    def _process_video(self, images: VideoNumpy, num_frm: int = MAX_NUMBER_OF_FRAMES):
        if len(images) > num_frm:
            num_frm = min(num_frm, len(images))
            step_size = len(images) / (num_frm + 1)
            indices = [int(i * step_size) for i in range(num_frm)]
            images = [images[i] for i in indices]
        images = [Image.fromarray(arr) for arr in images]
        print(f"sample {len(images)} frames.")
        return images

    def __init__(self, max_size=MAX_IMAGE_SIZE):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_size = max_size
        # self._load_video = get_class_from_dynamic_module(
        #     "ixc_utils.load_video", self._HF_MODEL
        # )
        self._frame2img = get_class_from_dynamic_module(
            "ixc_utils.frame2img", self._HF_MODEL
        )
        self._video_transform = get_class_from_dynamic_module(
            "ixc_utils.Video_transform", self._HF_MODEL
        )
        self._get_font = get_class_from_dynamic_module(
            "ixc_utils.get_font", self._HF_MODEL
        )
        self._gen_config = GenerationConfig(top_k=0, top_p=0.8, temperature=0.1)
        engine_config = TurbomindEngineConfig(
            model_format="awq", cache_max_entry_count=0.2
        )
        self._pipe = pipeline(
            self._HF_MODEL + "-4bit", backend_config=engine_config, device=self.device
        )

    def __call__(
        self,
        image: Image.Image | None,
        video: VideoNumpy | None,
        prompt: str,
    ) -> str:
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            image = self._transform_image(image, self.max_size)
        elif video is not None:
            video = self._process_video(video)
            if len(video) > self.MAX_NUMBER_OF_FRAMES:
                raise ValueError("Video is too long")
            image = self._frame2img(video, self._get_font())
            image = self._video_transform(image)

        sess = self._pipe.chat((prompt, image), gen_config=self._gen_config)
        return sess.response.text