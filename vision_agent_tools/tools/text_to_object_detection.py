from typing import List, Any
from PIL import Image
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.models.model_registry import TextToObjectDetectionModel


class TextToObjectDetectionOutput(BaseModel):
    output: Any


class TextToObjectDetection(BaseTool):
    """
    Tool to perform object detection based on text prompts using a specified ML model
    """

    def __init__(self, model: str):
        self.model_name = model
        model_class = get_model_class(model_name=model, task=TextToObjectDetectionModel)
        model_instance = model_class()
        super().__init__(model=model_instance)

    def __call__(
        self, image: Image.Image, prompts: List[str]
    ) -> List[TextToObjectDetectionOutput]:
        """
        Run object detection on the image based on text prompts.

        Args:
            image (Image.Image): The input image for object detection.
            prompts (List[str]): List of text prompts for object detection.

        Returns:
            List[TextToObjectDetectionOutput]: A list of detection results for each prompt.
        """
        results = []

        for prompt in prompts:
            prediction = self.model(image=image, task=prompt)
            output = TextToObjectDetectionOutput(output=prediction)
            results.append(output)

        return results
