import torch
import numpy as np
from PIL import Image
from pydantic import validate_call

from qwen_vl_utils import process_vision_info
from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class Qwen2VL(BaseMLModel):
    """
    [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) is a tool that ...

    NOTE: The Qwen2-VL model should be used in GPU environments.
    """

    _HF_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    def _process_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        return image

    def _process_video(self, video: VideoNumpy) -> list[Image.Image]:
        shape = video.shape
        images = [
            self._process_image(Image.fromarray(video[t])) for t in range(shape[0])
        ]
        return images

    def __init__(self, device: Device | None = Device.GPU) -> None:
        """
        Initializes the Qwen2-VL model.
        """
        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device.value

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._HF_MODEL, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self._processor = AutoProcessor.from_pretrained(self._HF_MODEL)

        self._model.to(self.device)
        self._model.eval()

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | None = None,
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
    ) -> list[str]:
        """
        Qwen2-VL model answers questions about a video or image.

        Args:
            prompt (str): The prompt with the question to be answered.
            image (Image.Image | None): The image to be analyzed.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            chunk_length (int): The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.

        Returns:
            list[str]: The answers to the prompt.
        """
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")
        # create the conversation template
        if image is not None:
            if prompt is None:
                prompt = "Describe this image."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self._process_image(image)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        if video is not None:
            shape = video[0].shape
            if prompt is None:
                prompt = "Describe this video."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": self._process_video(video),
                            "max_pixels": shape[0] * shape[1],
                            "fps": 1.0,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        # process the inputs
        text_prompt = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        # run the inference
        output_ids = self._model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text

    def to(self, device: Device):
        self._model.to(device=device.value)
