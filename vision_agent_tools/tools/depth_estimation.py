from typing import Any, Dict
from enum import Enum
from PIL import Image
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class


class DepthEstimationModel(str, Enum):
    DEPTH_ANYTHING_V2 = "depth_anything_v2"


class DepthEstimation(BaseTool):
    def __init__(self, model: DepthEstimationModel):
        model_class = get_model_class(model_name=model)
        model_instance = model_class()
        super().__init__(model=model_instance())

    def __call__(
        self,
        image: Image.Image,
        **model_config: Dict[str, Any],
    ):
        """
        Run depth estimation detection on the image provided.

        Args:
            image (Image.Image): The input image for object detection.

        Returns:
            DepthEstimationOutput: A estimation of depth.
        """
        result = self.model(image=image, **model_config)
        return result
