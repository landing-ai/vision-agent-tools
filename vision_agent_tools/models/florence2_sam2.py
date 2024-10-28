import logging
from typing import Any

import torch
from PIL import Image
from typing_extensions import Self
from pydantic import BaseModel, Field, ConfigDict, model_validator

from vision_agent_tools.shared_types import (
    BaseMLModel,
    VideoNumpy,
    PromptTask,
    ODResponse,
)
from vision_agent_tools.models.sam2 import Sam2, Sam2Config
from vision_agent_tools.models.florence2 import Florence2, Florence2Config

_LOGGER = logging.getLogger(__name__)


class Florence2SAM2Config(BaseModel):
    sam2_config: Sam2Config | None = Sam2Config()
    florence2_config: Florence2Config | None = Florence2Config()


class Florence2Sam2Request(BaseModel):
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
        description="The non-maximum suppression threshold value used to filter the Florence2 predictions.",
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


class Florence2SAM2(BaseMLModel):
    """A class that receives a video or images, a text prompt and returns the instance
    segmentation based on the input for each frame.
    """

    def __init__(
        self, model_config: Florence2SAM2Config | None = Florence2SAM2Config()
    ):
        """
        Initializes the Florence2SAM2 object with a pre-trained Florence2 model
        and a SAM2 model.
        """
        self._model_config = model_config
        self._florence2 = Florence2(self._model_config.florence2_config)
        self._sam2 = Sam2(self._model_config.sam2_config)

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
    ) -> list[list[dict[str, Any]]]:
        """
        Florence2Sam2 model find objects in images and track objects in a video.

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
        Florence2Sam2Request(
            prompt=prompt,
            images=images,
            video=video,
            chunk_length_frames=chunk_length_frames,
            iou_threshold=iou_threshold,
            nms_threshold=nms_threshold,
        )

        florence2_payload = {
            "task": PromptTask.CAPTION_TO_PHRASE_GROUNDING,
            "prompt": prompt,
            "images": images,
            "video": video,
            "batch_size": 5,
            "nms_threshold": nms_threshold,
        }
        if video is not None:
            florence2_payload["chunk_length_frames"] = chunk_length_frames

        florence2_response = self._florence2(**florence2_payload)
        od_response = [
            ODResponse(**item) if item is not None else None
            for item in florence2_response
        ]

        if images is not None:
            return self._sam2(
                images=images,
                bboxes=od_response,
            )

        if video is not None:
            return self._sam2(
                video=video,
                bboxes=od_response,
                chunk_length_frames=chunk_length_frames,
                iou_threshold=iou_threshold,
            )

    def load_base(self) -> None:
        """Load the base Florence-2 model."""
        self._florence2.load_base()

    def fine_tune(self, checkpoint: str) -> None:
        """Load the fine-tuned Florence-2 model."""
        self._florence2.fine_tune(checkpoint)
