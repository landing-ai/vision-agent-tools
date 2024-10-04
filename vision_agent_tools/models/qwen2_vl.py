import torch
from PIL import Image
from pydantic import BaseModel, validate_call, Field

from qwen_vl_utils import process_vision_info
from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class Qwen2VLConfig(BaseModel):
    hf_model: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        description="Name of the model",
    )
    device: Device = Field(
        default=Device.GPU
        if torch.cuda.is_available()
        else Device.MPS
        if torch.backends.mps.is_available()
        else Device.CPU,
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. Default is the first available GPU.",
    )


class Qwen2VL(BaseMLModel):
    """
    [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) is a model that is capable
    of accurately identifying and comprehending the content within images,
    regardless of their clarity, resolution, or extreme aspect ratios.

    NOTE: The Qwen2-VL model should be used in GPU environments.
    """

    def _process_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        return image

    def _process_video(self, video: VideoNumpy) -> list[Image.Image]:
        shape = video.shape
        images = [
            self._process_image(Image.fromarray(video[t])) for t in range(shape[0])
        ]
        return images

    def __init__(self, model_config: Qwen2VLConfig | None = None) -> None:
        """
        Initializes the Qwen2-VL model.
        """
        self._model_config = model_config or Qwen2VLConfig()
        self.device = self._model_config.device

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_config.hf_model,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._processor = AutoProcessor.from_pretrained(self._model_config.hf_model)

        self._model.to(self.device)
        self._model.eval()

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | None = None,
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
    ) -> list[str]:
        """
        Qwen2-VL model answers questions about a video or image.

        Args:
            prompt (str): The prompt with the question to be answered.
            images (list[Image.Image]): A list of images for the model to process. None if using video.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            chunk_length (int): The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.

        Returns:
            list[str]: The answers to the prompt.
        """
        if images is None and video is None:
            raise ValueError("Either 'images' or 'video' must be provided.")
        if images is not None and video is not None:
            raise ValueError("Only one of 'images' or 'video' can be provided.")
        # create the conversation template
        if images is not None:
            if prompt is None:
                prompt = "Describe this image."
            images_input = [
                {"type": "image", "image": self._process_image(image)}
                for image in images
            ]
            conversation = [
                {
                    "role": "user",
                    "content": images_input + [{"type": "text", "text": prompt}],
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
