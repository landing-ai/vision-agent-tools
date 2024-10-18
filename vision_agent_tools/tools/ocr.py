from enum import Enum
from typing import Any

from PIL import Image

from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.shared_types import BaseTool, PromptTask, Device


class OCRModel(str, Enum):
    FLORENCE2 = "florence2"


class OCR(BaseTool):
    """Tool to perform OCR on images using a specified ML model"""
    def __init__(self, model: OCRModel = OCRModel.FLORENCE2):
        model_class = get_model_class(model_name=model.value)
        model_instance = model_class()
        super().__init__(model=model_instance())
        self._ocr_tasks = [PromptTask.OCR, PromptTask.OCR_WITH_REGION]

    def __call__(
        self, images: list[Image.Image], task: PromptTask = PromptTask.OCR
    ) -> dict[str, Any]:
        if task not in self._ocr_tasks:
            raise ValueError(f"Invalid task: {task}. Supported tasks are: {self._ocr_tasks}")
        return self.model(images=images, task=task)

    def to(self, device: Device) -> None:
        raise NotImplementedError("This method is not supported for OCR tool.")
