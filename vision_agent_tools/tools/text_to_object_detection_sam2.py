import logging
from typing import Any

import torch
from PIL import Image
from typing_extensions import Self
from pydantic import BaseModel, Field, ConfigDict, model_validator

from vision_agent_tools.shared_types import (
    BaseMLModel,
    VideoNumpy,
    ODWithScoreResponse,
)
from vision_agent_tools.models.sam2 import Sam2, Sam2Config
from vision_agent_tools.models.florence2 import Florence2Config
from vision_agent_tools.models.owlv2 import OWLV2Config
from vision_agent_tools.tools.text_to_object_detection import (
    TextToObjectDetection,
    TextToObjectDetectionModel,
)

_LOGGER = logging.getLogger(__name__)


class Text2ODSAM2Config(BaseModel):
    sam2_config: Sam2Config | None = None
    text2od_config: Florence2Config | OWLV2Config | None = None


class Text2ODSam2Request(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: list[str] | None = Field(
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
    confidence: float = Field(
        1.0,
        ge=0.1,
        le=1.0,
        description="The confidence threshold for model predictions.",
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


class Text2ODSAM2(BaseMLModel):
    """A class that receives a video or images, a text prompt and returns the instance
    segmentation based on the input for each frame.
    """

    def __init__(
        self,
        model: TextToObjectDetectionModel = TextToObjectDetectionModel.OWLV2,
        model_config: Text2ODSAM2Config | None = None,
    ):
        """
        Initializes the Text2ODSAM2 object with a pre-trained text2od model
        and a SAM2 model.
        """
        self._model = model
        self._model_config = model_config or Text2ODSAM2Config()
        self._text2od = TextToObjectDetection(
            model=model, model_config=self._model_config.text2od_config
        )
        self._sam2 = Sam2(self._model_config.sam2_config)

    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
        *,
        chunk_length_frames: int | None = 20,
        iou_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        confidence: float = 0.1,
    ) -> list[list[dict[str, Any]]]:
        """
        Text2ODSam2 model find objects in images and track objects in a video.

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
                Text2OD predictions.
            confidence:
                Confidence threshold for model predictions

        Returns:
            list[list[dict[str, Any]]]:
                A list where each item represents each frames predictions.
                [[{
                    "id": 0,
                    "mask": rle,
                    "label": "car",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "score": 0.55,
                    "logits": None,
                }]]
        """
        Text2ODSam2Request(
            prompts=prompts,
            images=images,
            video=video,
            chunk_length_frames=chunk_length_frames,
            iou_threshold=iou_threshold,
            nms_threshold=nms_threshold,
            confidence=confidence,
        )

        text2od_payload = {
            "prompts": prompts,
            "images": images,
            "video": video,
            "nms_threshold": nms_threshold,
        }
        if video is not None and self._model is TextToObjectDetectionModel.FLORENCE2:
            text2od_payload["chunk_length_frames"] = chunk_length_frames
        if confidence is not None:
            text2od_payload["confidence"] = confidence

        text2od_payload_response = self._text2od(**text2od_payload)
        od_response = [
            ODWithScoreResponse(**item)
            if item is not None and len(item.get("labels")) > 0
            else None
            for item in text2od_payload_response
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
                confidence=confidence,
            )
