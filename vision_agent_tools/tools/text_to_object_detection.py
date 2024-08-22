from pydantic import BaseModel
from typing import List
from PIL import Image
from vision_agent_tools.tools.base_tool import Tool
from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.shared_types import BaseMLModel


class BoundingBox(BaseModel):
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class TextToObjectDetectionOutput(BaseModel):
    tasks: str
    bboxes: List[BoundingBox]


class TextToObjectDetection(Tool):
    """
    Perform object detection from text tasks
    """

    def __init__(self, model: str):
        model_class = get_model_class(model)
        self.model: BaseMLModel = model_class()

    def run(
        self, image: Image.Image, tasks: List[str]
    ) -> List[TextToObjectDetectionOutput]:
        """
        Run object detection on the image based on text tasks

        Args:
            image (Image.Image): The input image for object detection.
            task (PromptTask): The task to be performed on the image.
        """
        results = []

        for tasks in tasks:
            prediction = self.model.predict(
                image=image, task="object_detection", tasks=tasks
            )
            output = TextToObjectDetectionOutput(
                tasks=tasks,
                bboxes=[BoundingBox(**bbox) for bbox in prediction["bboxes"]],
            )
            results.append(output)

        return results
