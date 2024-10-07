import torch
from PIL import Image
from typing import Annotated
from pydantic import BaseModel, validate_call, Field
from annotated_types import Len


from qwen_vl_utils import process_vision_info
from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MAX_NUMBER_OF_IMAGES = 10
MAX_NUMBER_OF_FRAMES = 24

Images = Annotated[
    list[Image.Image], Len(min_length=1, max_length=MAX_NUMBER_OF_IMAGES)
]
Frames = Annotated[int, Field(ge=1, le=MAX_NUMBER_OF_FRAMES)]


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

    _IMAGE_MAX_SHAPE_VALUE = (960, 600)
    _IMAGE_MAX_PIXELS = _IMAGE_MAX_SHAPE_VALUE[0] * _IMAGE_MAX_SHAPE_VALUE[1]
    _VIDEO_MAX_SHAPE_VALUE = (560, 560)
    _VIDEO_MAX_PIXELS = _VIDEO_MAX_SHAPE_VALUE[0] * _VIDEO_MAX_SHAPE_VALUE[1]

    def _process_image(self, image: Image.Image, max_shape: tuple) -> Image.Image:
        image = image.convert("RGB")
        if image.size[0] > max_shape[0] or image.size[1] > max_shape[1]:
            image.thumbnail(max_shape)
        return image

    def _process_video(self, frames: VideoNumpy, num_frames: int) -> list[Image.Image]:
        if len(frames) > num_frames:
            num_frames = min(num_frames, len(frames))
            step_size = len(frames) / (num_frames + 1)
            indices = [int(i * step_size) for i in range(num_frames)]
            frames = [frames[i] for i in indices]
        frames = [
            self._process_image(Image.fromarray(arr), self._VIDEO_MAX_SHAPE_VALUE)
            for arr in frames
        ]
        return frames

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
            attn_implementation="flash_attention_2",
        )
        self._processor = AutoProcessor.from_pretrained(self._model_config.hf_model)

        self._model.to(self.device)
        self._model.eval()

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | None = None,
        images: Images | None = None,
        video: VideoNumpy | None = None,
        frames: Frames = MAX_NUMBER_OF_FRAMES,
    ) -> list[str]:
        """
        Qwen2-VL model answers questions about a video or image.

        Args:
            prompt (str): The prompt with the question to be answered.
            images (list[Image.Image]): A list of images for the model to process. None if using video.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            frames (int): The number of frames to be used from the video.

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
                {
                    "type": "image",
                    "image": self._process_image(image, self._IMAGE_MAX_SHAPE_VALUE),
                }
                for image in images
            ]
            conversation = [
                {
                    "role": "user",
                    "content": images_input + [{"type": "text", "text": prompt}],
                    "max_pixels": self._IMAGE_MAX_PIXELS,
                }
            ]
        if video is not None:
            if prompt is None:
                prompt = "Describe this video."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": self._process_video(video, frames),
                            "max_pixels": self._VIDEO_MAX_PIXELS,
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
        output_ids = self._model.generate(**inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        torch.cuda.empty_cache()
        return output_text

    def to(self, device: Device):
        self._model.to(device=device.value)
