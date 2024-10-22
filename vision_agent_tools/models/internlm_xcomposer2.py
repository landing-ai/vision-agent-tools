from typing import Annotated

import torch
from PIL import Image
from pydantic import Field, validate_call
from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

from vision_agent_tools.models.utils import get_device
from vision_agent_tools.shared_types import BaseMLModel, VideoNumpy
from vision_agent_tools.helpers.ixc_utils import frame2img, Video_transform, get_font


MAX_NUMBER_OF_FRAMES = 32

Frames = Annotated[int, Field(ge=1, le=MAX_NUMBER_OF_FRAMES)]


class InternLMXComposer2(BaseMLModel):
    """
    [InternLM-XComposer-2.5](https://huggingface.co/internlm/internlm-xcomposer2d5-7b-4bit) is a tool that excels in various text-image
    comprehension and composition applications, achieving GPT-4V level capabilities.

    NOTE: The InternLM-XComposer-2.5 model should be used in GPU environments.
    """

    _HF_MODEL = "internlm/internlm-xcomposer2d5-7b"
    _MAX_NUMBER_OF_FRAMES = MAX_NUMBER_OF_FRAMES
    _MAX_IMAGE_SIZE = 1024

    def __init__(self) -> None:
        """
        Initializes the InternLMXComposer2.5 model.
        """
        self.device = get_device()
        self._gen_config = GenerationConfig(top_k=0, top_p=0.8, temperature=0.1)
        engine_config = TurbomindEngineConfig(
            model_format="awq", cache_max_entry_count=0.2
        )
        self._pipe = pipeline(
            self._HF_MODEL + "-4bit",
            backend_config=engine_config,
            device=self.device.value,
        )

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        frames: Frames = MAX_NUMBER_OF_FRAMES,
        chunk_length: int | None = None,
    ) -> list[str]:
        """
        InternLMXComposer2 model answers questions about a video or image.

        Args:
            prompt (str): The prompt with the question to be answered.
            image (Image.Image | None): The image to be analyzed.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            frames (int): The number of frames to be used from the video.
            chunk_length (int): The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.

        Returns:
            list[str]: The answers to the prompt.
        """
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            media = self._transform_image(image)
            sess = self._pipe.chat((prompt, media), gen_config=self._gen_config)
            return [sess.response.text]
        if video is not None:
            num_frames = video.shape[0]
            if chunk_length is None:
                chunk_length = num_frames
            answers: list[str] = []
            for i in range(0, num_frames, chunk_length):
                chunk = video[i : i + chunk_length, :, :, :]
                chunk = self._process_video(chunk, frames)
                image_frames = frame2img(chunk, get_font())
                media = Video_transform(image_frames)
                sess = self._pipe.chat((prompt, media), gen_config=self._gen_config)
                response = sess.response.text
                answers.append(response)
            return answers

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
