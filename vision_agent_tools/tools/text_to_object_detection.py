import logging
from enum import Enum
from typing import Any

from PIL import Image
from typing_extensions import Self
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vision_agent_tools.shared_types import PromptTask
from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.shared_types import (
    BaseTool,
    Device,
    VideoNumpy,
)
from vision_agent_tools.models.owlv2 import OWLV2Config
from vision_agent_tools.models.florence2 import Florence2Config

_LOGGER = logging.getLogger(__name__)


class TextToObjectDetectionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: list[str] = Field(
        description="The prompt to be used for object detection."
    )
    images: list[Image.Image] | None = None
    video: VideoNumpy | None = None
    nms_threshold: float = Field(
        0.3,
        ge=0.1,
        le=1.0,
        description="The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for model predictions",
    )

    @model_validator(mode="after")
    def check_images_and_video(self) -> Self:
        if self.video is None and self.images is None:
            raise ValueError("video or images is required")

        if self.video is not None and self.images is not None:
            raise ValueError("Only one of them are required: video or images")

        return self


class TextToObjectDetectionModel(str, Enum):
    OWLV2 = "owlv2"
    FLORENCE2 = "florence2"


class TextToObjectDetection(BaseTool):
    """Tool to perform object detection based on text prompts using a specified ML model"""

    def __init__(
        self,
        model: TextToObjectDetectionModel = TextToObjectDetectionModel.OWLV2,
        model_config: OWLV2Config | None = None,
    ):
        self.model_name = model

        self.model_class = get_model_class(model_name=model.value)
        model_instance = self.model_class()

        if model is TextToObjectDetectionModel.OWLV2:
            self.model_config = model_config or OWLV2Config()
            super().__init__(model=model_instance(self.model_config))
        elif model is TextToObjectDetectionModel.FLORENCE2:
            self.model_config = model_config or Florence2Config()
            super().__init__(model=model_instance(self.model_config))

    def __call__(
        self,
        prompts: list[str],
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
        *,
        nms_threshold: float = 0.3,
        confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run object detection on the image based on text prompts.

        Args:
            prompts:
                The prompt to be used for object detection.
            images:
                The images to be analyzed.
            video:
                A numpy array containing the different images, representing the video.
            nms_threshold:
                The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).
            confidence:
                Confidence threshold for model predictions.

        Returns:
            list[dict[str, Any]]:
                A list of detection results for the prompts.
        """
        TextToObjectDetectionRequest(
            prompts=prompts,
            images=images,
            video=video,
            nms_threshold=nms_threshold,
            confidence=confidence,
        )

        if self.model_name is TextToObjectDetectionModel.OWLV2:
            payload = {
                "prompts": prompts,
                "images": images,
                "video": video,
                "nms_threshold": nms_threshold,
            }
            if confidence is not None:
                payload["confidence"] = confidence

            return self.model(**payload)

        if self.model_name is TextToObjectDetectionModel.FLORENCE2:
            if confidence is not None:
                _LOGGER.warning(
                    "Confidence threshold is not supported for Florence2 model."
                )

            task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
            prompt = ", ".join(prompts)
            return self.model(
                task,
                prompt=prompt,
                images=images,
                video=video,
                nms_threshold=nms_threshold,
            )

    def to(self, device: Device):
        raise NotImplementedError(
            "This method is not supported for TextToObjectDetection tool."
        )
