from typing import List, Any
from enum import Enum
from PIL import Image
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class


class TextToObjectDetectionOutput(BaseModel):
    output: Any


class TextToObjectDetectionModel(str, Enum):
    FLORENCEV2 = "florencev2"


class TextToObjectDetection(BaseTool):
    """
    Tool to perform object detection based on text prompts using a specified ML model
    """

    def __init__(self, model: TextToObjectDetectionModel):
        if model not in TextToObjectDetectionModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )
        model_class = get_model_class(model_name=model)
        model_instance = model_class()
        super().__init__(model=model_instance)

    def __call__(
        self, image: Image.Image, task: List[str]
    ) -> List[TextToObjectDetectionOutput]:
        prediction = self.model(image=image, task=task)
        output = TextToObjectDetectionOutput(output=prediction)

        return output
