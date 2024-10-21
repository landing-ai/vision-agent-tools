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


class TextToObjectDetectionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str | None = Field(
        "", description="The text input that complements the prompt task."
    )
    images: list[Image.Image] | None = None
    video: VideoNumpy | None = None
    nms_threshold: float = Field(
        1.0,
        ge=0.1,
        le=1.0,
        description="The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).",
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
            super().__init__(model=model_instance())

    def __call__(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
        *,
        nms_threshold: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Run object detection on the image based on text prompts.

        Args:
            prompt:
                The text input that complements the media to find or track objects.
            images:
                The images to be analyzed.
            video:
                A numpy array containing the different images, representing the video.
            nms_threshold:
                The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).

        Returns:
            list[dict[str, Any]]:
                A list of detection results for the prompt.
        """
        TextToObjectDetectionRequest(
            prompt=prompt, images=images, video=video, nms_threshold=nms_threshold
        )

        if self.model_name is TextToObjectDetectionModel.OWLV2:
            return self.model(prompt, images=images, video=video)

        if self.model_name is TextToObjectDetectionModel.FLORENCE2:
            task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
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
