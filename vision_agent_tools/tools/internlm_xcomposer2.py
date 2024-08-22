import torch
from PIL import Image
from vision_agent_tools.shared_types import BaseTool, VideoNumpy
from pydantic import Field, validate_call
from typing import Annotated

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from transformers.dynamic_module_utils import get_class_from_dynamic_module


MAX_NUMBER_OF_FRAMES = 32

Frames = Annotated[int, Field(ge=1, le=MAX_NUMBER_OF_FRAMES)]


class InternLMXComposer2(BaseTool):
    """
    [InternLM-XComposer-2.5](https://huggingface.co/internlm/internlm-xcomposer2d5-7b-4bit) is a tool that excels in various text-image
    comprehension and composition applications, achieving GPT-4V level capabilities.

    NOTE: The InternLM-XComposer-2.5 model should be used in GPU environments.
    """

    _HF_MODEL = "internlm/internlm-xcomposer2d5-7b"
    _MAX_NUMBER_OF_FRAMES = MAX_NUMBER_OF_FRAMES
    _MAX_IMAGE_SIZE = 1024

    def _transform_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if image.size[0] > self._MAX_IMAGE_SIZE or image.size[1] > self._MAX_IMAGE_SIZE:
            image.thumbnail((self._MAX_IMAGE_SIZE, self._MAX_IMAGE_SIZE))
        return image

    def _process_video(self, images: VideoNumpy, num_frames: int) -> list[Image.Image]:
        if len(images) > num_frames:
            num_frames = min(num_frames, len(images))
            step_size = len(images) / (num_frames + 1)
            indices = [int(i * step_size) for i in range(num_frames)]
            images = [images[i] for i in indices]
        images = [self._transform_image(Image.fromarray(arr)) for arr in images]
        return images

    def __init__(self) -> None:
        """
        Initializes the InternLMXComposer2.5 model.
        """
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
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

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        frames: Frames = MAX_NUMBER_OF_FRAMES,
    ) -> str:
        """
        InternLMXComposer2 model answers questions about a video or image.

        Args:
            prompt (str): The prompt with the question to be answered.
            image (Image.Image | None): The image to be analyzed.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            frames (int): The number of frames to be used from the video.

        Returns:
            str: The answer to the prompt.
        """
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            media = self._transform_image(image)
        if video is not None:
            video = self._process_video(video, frames)
            image_frames = self._frame2img(video, self._get_font())
            media = self._video_transform(image_frames)

        sess = self._pipe.chat((prompt, media), gen_config=self._gen_config)
        return sess.response.text
