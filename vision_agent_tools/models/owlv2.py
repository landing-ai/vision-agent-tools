import logging
from typing import Tuple

import torch
import numpy as np
from PIL import Image
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator, ConfigDict
from transformers.utils import TensorType
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.image_transforms import center_to_corners_format
from transformers.models.owlv2.image_processing_owlv2 import box_iou
from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput

from vision_agent_tools.helpers.filters import filter_bbox_predictions
from vision_agent_tools.models.utils import get_device
from vision_agent_tools.shared_types import (
    BaseMLModel,
    Device,
    VideoNumpy,
    ODWithScoreResponse,
)

_LOGGER = logging.getLogger(__name__)


class OWLV2Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(
        default="google/owlv2-large-patch14-ensemble",
        description="Name of the model",
    )
    device: Device = Field(
        default=get_device(),
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. "
        "Default is the first available GPU.",
    )


# TODO: fix batch_size bigger than 1, for some cases it's not working
# it raises the error "RuntimeError: shape '[2, 0, 768]' is invalid for input of size 768"
# location: `query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])`
# transformers/models/owlv2/modeling_owlv2.py, line 1697, in forward
class Owlv2Request(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: list[str] = Field(
        description="The prompt to be used for object detection."
    )
    images: list[Image.Image] | None = None
    video: VideoNumpy | None = None
    batch_size: int = Field(
        1,
        ge=1,
        description="The batch size used for processing multiple images or video frames.",
    )
    nms_threshold: float = Field(
        0.3,
        ge=0.1,
        le=1.0,
        description="The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).",
    )
    confidence: float = Field(
        default=0.1,
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


class Owlv2(BaseMLModel):
    """
    Tool for object detection using the pre-trained Owlv2 model from
    [Transformers](https://github.com/huggingface/transformers).

    This tool takes images and a prompt as input, performs object detection using
    the Owlv2 model, and returns a list of objects containing the predicted labels,
    confidence scores, and bounding boxes for detected objects with confidence
    exceeding a threshold.
    """

    def __init__(self, model_config: OWLV2Config | None = OWLV2Config()):
        """Loads the pre-trained Owlv2 processor and model from Transformers."""
        self.model_config = model_config
        self._model = Owlv2ForObjectDetection.from_pretrained(
            self.model_config.model_name
        )
        self._processor = Owlv2ProcessorWithNMS.from_pretrained(
            self.model_config.model_name
        )
        self._model.to(self.model_config.device)
        self._model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        images: list[Image.Image] | None = None,
        video: VideoNumpy[np.uint8] | None = None,
        *,
        batch_size: int = 1,
        nms_threshold: float = 0.3,
        confidence: float = 0.1,
    ) -> list[ODWithScoreResponse]:
        """Performs object detection on images using the Owlv2 model.

        Args:
            prompts:
                The prompt to be used for object detection.
            images:
                The images to be analyzed.
            video:
                A numpy array containing the different images, representing the video.
            batch_size:
                The batch size used for processing multiple images or video frames.
            nms_threshold:
                The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).
            confidence:
                Confidence threshold for model predictions.

        Returns:
            list[ODWithScoreResponse]:
                A list of `ODWithScoreResponse` objects containing the predicted
                labels, confidence scores, and bounding boxes for detected objects
                with confidence exceeding the threshold. The item will be None if
                no objects are detected above the confidence threshold for an specific
                image / frame.
        """
        Owlv2Request(
            prompts=prompts,
            images=images,
            video=video,
            batch_size=batch_size,
            nms_threshold=nms_threshold,
            confidence=confidence,
        )

        if images is not None:
            images = [image.convert("RGB") for image in images]

        if video is not None:
            images = [Image.fromarray(frame).convert("RGB") for frame in video]

        return self._run_inference(
            prompts,
            images,
            batch_size=batch_size,
            confidence=confidence,
            nms_threshold=nms_threshold,
        )

    def to(self, device: Device) -> None:
        raise NotImplementedError("This method is not supported for Owlv2 model.")

    def _run_inference(
        self,
        prompts: list[str],
        images: list[Image.Image],
        *,
        batch_size: int,
        confidence: float,
        nms_threshold: float,
    ) -> list[ODWithScoreResponse]:
        results = []
        for idx in range(0, len(images), batch_size):
            end_frame = idx + batch_size
            if end_frame >= len(images):
                end_frame = len(images)

            _LOGGER.info(
                f"Processing chunk, start frame: {idx}, end frame: {end_frame - 1}"
            )
            images_chunk = images[idx:end_frame]

            inputs = self._processor(
                text=prompts,
                images=images_chunk,
                return_tensors="pt",
                truncation=True,
            ).to(self.model_config.device)

            # Forward pass
            with torch.autocast(self.model_config.device):
                outputs = self._model(**inputs)

            target_sizes = [image.size[::-1] for image in images_chunk]
            results.extend(
                self._processor.post_process_object_detection_with_nms(
                    outputs=outputs,
                    threshold=confidence,
                    nms_threshold=nms_threshold,
                    target_sizes=target_sizes,
                )
            )

        filtered_bboxes_all_frames = []
        for result, image in zip(results, images):
            result["bboxes"] = result.pop("boxes")
            filtered_bboxes = filter_bbox_predictions(
                result, image.size, nms_threshold=nms_threshold
            )
            filtered_bboxes["labels"] = [
                prompts[label] for label in filtered_bboxes["labels"]
            ]
            filtered_bboxes_all_frames.append(filtered_bboxes)

        return filtered_bboxes_all_frames


class Owlv2ProcessorWithNMS(Owlv2Processor):
    def post_process_object_detection_with_nms(
        self,
        outputs: OwlViTObjectDetectionOutput,
        *,
        threshold: float = 0.1,
        nms_threshold: float = 0.3,
        target_sizes: TensorType | list[Tuple] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Converts the raw output of [`OwlViTForObjectDetection`] into final
        bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.

        Args:
            outputs:
                Raw outputs of the model.
            threshold:
                Score threshold to keep object detection predictions.
            nms_threshold:
                IoU threshold to filter overlapping objects the raw detections.
            target_sizes:
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`)
                containing the target size `(height, width)` of each image in the batch.
                If unset, predictions will not be resized.
        Returns:
            `list[dict]`:
                A list of dictionaries, each dictionary containing the scores, labels
                and boxes for an image in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch "
                    "dimension of the logits"
                )

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Apply non-maximum suppression (NMS)
        # borrowed the implementation from HuggingFace Owlv2 post_process_image_guided_detection()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/image_processing_owlv2.py#L563-L573
        if nms_threshold < 1.0:
            for idx in range(boxes.shape[0]):
                for i in torch.argsort(-scores[idx]):
                    if not scores[idx][i]:
                        continue
                    ious = box_iou(boxes[idx][i, :].unsqueeze(0), boxes[idx])[0][0]
                    ious[i] = -1.0  # Mask self-IoU.
                    scores[idx][ious > nms_threshold] = 0.0

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            # Rescale coordinates, image is padded to square for inference,
            # that is why we need to scale boxes to the max size
            size = torch.max(img_h, img_w)
            scale_fct = torch.stack([size, size, size, size], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for score_array, label_array, box_array in zip(scores, labels, boxes):
            high_score_mask = score_array > threshold
            filtered_scores = score_array[high_score_mask]
            filtered_labels = label_array[high_score_mask]
            filtered_boxes = box_array[high_score_mask]

            results.append(
                {
                    "scores": filtered_scores.cpu().tolist(),
                    "labels": filtered_labels.cpu().tolist(),
                    "boxes": filtered_boxes.cpu().tolist(),
                }
            )

        return results
