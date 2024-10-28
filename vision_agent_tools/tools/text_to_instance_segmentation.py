from enum import Enum
from typing import Any

import torch
from PIL import Image
from typing_extensions import Self
from pydantic import BaseModel, Field, ConfigDict, model_validator

from vision_agent_tools.shared_types import BaseTool, VideoNumpy
from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.models.florence2_sam2 import Florence2SAM2Config


class TextToInstanceSegmentationModel(str, Enum):
    FLORENCE2SAM2 = "florence2sam2"


class TextToInstanceSegmentationRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str | None = Field(
        "",
        description="The text input that complements the media to find or track objects.",
    )
    images: list[Image.Image] | None = Field(
        None, description="The images to be analyzed."
    )
    video: VideoNumpy | None = Field(
        None,
        description="A numpy array containing the different images, representing the video.",
    )
    chunk_length_frames: int = Field(
        20,
        ge=1,
        description="The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.",
    )
    iou_threshold: float = Field(
        0.6,
        ge=0.1,
        le=1.0,
        description="The IoU threshold value used to compare last_predictions and new_predictions objects.",
    )
    nms_threshold: float = Field(
        1.0,
        ge=0.1,
        le=1.0,
        description="The non-maximum suppression threshold value used to filter the predictions.",
    )

    @model_validator(mode="after")
    def check_images_and_video(self) -> Self:
        if self.video is None and self.images is None:
            raise ValueError("video or images is required")

        if self.video is not None and self.images is not None:
            raise ValueError("Only one of them are required: video or images")

        if self.video is not None:
            if self.video.ndim != 4:
                raise ValueError("Video should have 4 dimensions")

        return self


class TextToInstanceSegmentationTool(BaseTool):
    """A tool that processes a video or images with text prompts for detection and segmentation."""

    def __init__(
        self,
        model: TextToInstanceSegmentationModel = TextToInstanceSegmentationModel.FLORENCE2SAM2,
        model_config: Florence2SAM2Config | None = Florence2SAM2Config(),
    ):
        self._model_name = model
        model_class = get_model_class(model_name=model.value)
        model_instance = model_class()
        super().__init__(model=model_instance(model_config))

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
        *,
        chunk_length_frames: int | None = 20,
        iou_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """TextToInstanceSegmentationTool model find segments in images and track objects in a video.

        Args:
            prompt:
                The text input that complements the media to find or track objects.
            images:
                The images to be analyzed.
            video:
                A numpy array containing the different images, representing the video.
            chunk_length_frames:
                The number of frames for each chunk of video to analyze.
                The last chunk may have fewer frames.
            iou_threshold:
                The IoU threshold value used to compare last_predictions and
                new_predictions objects.
            nms_threshold:
                The non-maximum suppression threshold value used to filter the
                Florence2 predictions.

        Returns:
            list[list[dict[str, Any]]]:
                A list where each item represents each frames predictions.
                [[{
                    "id": 0,
                    "mask": rle,
                    "label": "car",
                    "bbox": [0.1, 0.2, 0.3, 0.4]
                }]]
        """
        TextToInstanceSegmentationRequest(
            prompt=prompt,
            images=images,
            video=video,
            chunk_length_frames=chunk_length_frames,
            iou_threshold=iou_threshold,
            nms_threshold=nms_threshold,
        )

        return self.model(
            prompt=prompt,
            images=images,
            video=video,
            chunk_length_frames=chunk_length_frames,
            iou_threshold=iou_threshold,
            nms_threshold=nms_threshold,
        )

    def load_base(self) -> None:
        """Load the base model."""
        if self._model_name != TextToInstanceSegmentationModel.FLORENCE2SAM2:
            raise NotImplementedError(
                f"Loading base model is not supported for {self._model_name}"
            )

        self.model.load_base()

    def fine_tune(self, checkpoint: str) -> None:
        """Load the fine-tuned model."""
        if self._model_name != TextToInstanceSegmentationModel.FLORENCE2SAM2:
            raise NotImplementedError(
                f"Loading fine-tuned model is not supported for {self._model_name}"
            )

        self.model.fine_tune(checkpoint)
