from typing import List, Any
from enum import Enum

import numpy as np
from PIL import Image
from pydantic import BaseModel

from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.shared_types import BaseTool, VideoNumpy

from vision_agent_tools.models.owlv2 import OWLV2Config


class TextToObjectDetectionOutput(BaseModel):
    output: Any


class TextToObjectDetectionModel(str, Enum):
    OWLV2 = "owlv2"


class TextToObjectDetection(BaseTool):
    """
    Tool to perform object detection based on text prompts using a specified ML model
    """

    def __init__(
        self, model: TextToObjectDetectionModel, model_config: OWLV2Config | None = None
    ):
        if model not in TextToObjectDetectionModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )
        model_class = get_model_class(model_name=model)

        if model_config is None:
            model_config = OWLV2Config()

        self.model_config = model_config
        super().__init__(model=model_class())

    def __call__(
        self,
        prompts: List[str],
        image: Image.Image | None = None,
        video: VideoNumpy[np.uint8] | None = None,
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

        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            prediction = self.model(
                image=image, prompts=prompts, model_config=self.model_config
            )
        if video is not None:
            prediction = self.model(
                video=video, prompts=prompts, model_config=self.model_config
            )

        output = TextToObjectDetectionOutput(output=prediction)
        results.append(output)

        return results
