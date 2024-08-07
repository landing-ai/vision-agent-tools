import torch
from PIL import Image
from pydantic import validate_call
from vision_agent_tools.types import VideoNumpy
from vision_agent_tools.tools.shared_types import BaseTool

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class InternLMXComposer2(BaseTool):
    _HF_MODEL = "internlm/internlm-xcomposer2d5-7b"

    def _transform_image(image: Image.Image, max_size: int):
        image = image.convert("RGB")
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size))
        return image

    def __init__(self, max_size=1024):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_size = max_size
        self._load_video = get_class_from_dynamic_module(
            "ixc_utils.load_video", self._HF_MODEL
        )
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

    # @validate_call(config={"arbitrary_types_allowed": True})
    def __call__(
        self,
        media: Image.Image | VideoNumpy,
        prompt: str,
    ) -> str:
        if isinstance(media, Image.Image):
            image = self._transform_image(media, self.max_size)
        elif isinstance(media, VideoNumpy):
            video = self._load_video(media)
            video = [self._transform_image(image, self.max_size) for image in video]
            # 32 frames is a ~60s video clip
            if len(video) > 32:
                raise ValueError("Video is too long")
            image = self._frame2img(video, self._get_font())
            image = self._video_transform(image)

        sess = self._pipe.chat((prompt, image), gen_config=self._gen_config)
        return sess.response.text
