from typing import List, Any
from enum import Enum
from PIL import Image
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class


class OCROutput(BaseModel):
    output: Any


class OCRModel(str, Enum):
    FLORENCEV2 = "florencev2"


class OCR(BaseTool):
    """
    Tool to perform OCR on an image using a specified ML model
    """

    def __init__(self, model: OCRModel):
        if model not in OCRModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )
        model_class = get_model_class(model_name=model)
        model_instance = model_class()
        super().__init__(model=model_instance)

    def __call__(
        self, image: Image.Image, task: List[str] = "<OCR>"
    ) -> List[OCROutput]:
        result = self.model(image=image, task=task)
        return result[task]
