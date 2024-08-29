from enum import Enum
from typing import Any, Optional, List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_NAME = "microsoft/Florence-2-large"
PROCESSOR_NAME = "microsoft/Florence-2-large"


class PromptTask(str, Enum):
    """
    Valid task_prompts options for the Florence2 model.

    """

    CAPTION = "<CAPTION>"
    """"""
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    """"""
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    """"""
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    """"""
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    """"""
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """"""
    OBJECT_DETECTION = "<OD>"
    """"""
    OCR = "<OCR>"
    """"""
    OCR_WITH_REGION = "<OCR_WITH_REGION>"
    """"""
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    """"""
    REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    """"""
    REGION_TO_SEGMENTATION = "<REGION_TO_SEGMENTATION>"
    """"""
    REGION_TO_CATEGORY = "<REGION_TO_CATEGORY>"
    """"""
    REGION_TO_DESCRIPTION = "<REGION_TO_DESCRIPTION>"
    """"""


class Florencev2(BaseMLModel):
    """
    [Florence-2](https://huggingface.co/microsoft/Florence-2-base) can interpret simple
    text prompts to perform tasks like captioning, object detection, and segmentation.

    NOTE: The Florence-2 model can only be used in GPU environments.
    """

    def _process_image(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    def _process_video(self, images: VideoNumpy) -> list[Image.Image]:
        return [self._process_image(Image.fromarray(arr)) for arr in images]

    def __init__(self):
        """
        Initializes the Florence-2 model.
        """
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            PROCESSOR_NAME, trust_remote_code=True
        )
        self._frame2img = get_class_from_dynamic_module(
            "ixc_utils.frame2img", self._HF_MODEL
        )
        self._video_transform = get_class_from_dynamic_module(
            "ixc_utils.Video_transform", self._HF_MODEL
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        task: PromptTask,
        image: Optional[Image.Image] = None,
        video: Optional[VideoNumpy] = None,
        prompt: Optional[str] = "",
    ) -> Any:
        """
        Florence-2 model sequence-to-sequence architecture enables it to excel in both
        zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.
        For more examples and details, refer to the [Florence-2 sample usage](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb).

        Args:
            image (Image.Image): The input image for object detection.
            task (PromptTask): The task to be performed on the image.
            prompt (Optional[str]): The text input that complements the prompt task.

        Returns:
            Any: The output of the Florence-2 model based on the task and prompt.
        """
        if prompt is None:
            text_input = task
        else:
            text_input = task + prompt

        # Either video or image should be provided
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            image = self._process_image("RGB")
            return self._single_image_call(text_input, image, task, prompt)
        if video is not None:
            images = self._process_video(video)
            return [self._single_image_call(text_input, image, task, prompt) for image in images]


    def _single_image_call(self, text_input: str, image: Image.Image, task: PromptTask, prompt: str):
        inputs = self._processor(text=text_input, images=image, return_tensors="pt").to(
            self.device
        )

        with torch.autocast(self.device):
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=False,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self._processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )

        return parsed_answer

    def to(self, device: Device):
        self._model.to(device=device.value)

    def predict(
        self, image: Image.Image, prompts: Optional[List[str]] = None, **kwargs
    ) -> Any:
        task = kwargs.get("task", "")
        results = []
        for prompt in prompts:
            results.append(self.__call__(image=image, task=task, prompt=prompt))
        return results