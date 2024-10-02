import torch
from PIL import Image
from pydantic import validate_call

from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class Qwen2VL(BaseMLModel):
    """
    [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) is a tool that ...

    NOTE: The Qwen2-VL model should be used in GPU environments.
    """

    _HF_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

    def _process_image(self, image: Image.Image) -> Image.Image:
        pass

    def _process_video(self, video: VideoNumpy) -> list[Image.Image]:
        pass

    def __init__(self, device: Device | None = Device.GPU) -> None:
        """
        Initializes the Qwen2-VL model.
        """
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._HF_MODEL,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self._HF_MODEL)

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
        self._model.to(self.device)
        self._model.eval()

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | None = None,
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        chunk_length: int | None = None,
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

        if image is not None:
            if prompt is None:
                prompt = "Describe this image."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self._processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            output_ids = self._model.generate(**inputs, max_new_tokens=128)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print(output_text)
        if video is not None:
            return NotImplementedError("Video processing is not implemented yet.")

    def to(self, device: Device):
        self._model.to(device=device.value)
