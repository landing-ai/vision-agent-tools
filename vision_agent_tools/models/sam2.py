import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing_extensions import Self

from vision_agent_tools.models.utils import calculate_mask_iou, get_device
from vision_agent_tools.shared_types import (
    BaseMLModel,
    Device,
    ObjBboxAndMaskLabel,
    ObjMaskLabel,
    ODResponse,
    Sam2BitMask,
    VideoNumpy,
)

_LOGGER = logging.getLogger(__name__)


class Sam2Config(BaseModel):
    hf_model: str = Field(
        default="facebook/sam2-hiera-large",
        description="Name of the HuggingFace model",
    )
    device: Device = Field(
        default=get_device(),
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. "
        "Default is the first available GPU.",
    )


class Florence2Sam2Request(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: list[Image.Image] | None = Field(
        None, description="The images to be analyzed."
    )
    video: VideoNumpy | None = Field(
        None,
        description="A numpy array containing the different images, representing the video.",
    )
    bboxes: list[ODResponse] | None = Field(
        None,
        description="A list representing bboxes predictions for all frames.",
    )
    chunk_length_frames: int | None = 20
    input_box: np.ndarray | None = None
    input_points: np.ndarray | None = None
    input_label: np.ndarray | None = None
    multimask_output: bool = False
    iou_threshold: float = Field(
        0.6,
        ge=0.1,
        le=1.0,
        description="The IoU threshold value used to masks intersection.",
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


class Sam2(BaseMLModel):
    """It receives images, a prompt and returns the instance segmentation for the
    text prompt in each frame."""

    def __init__(self, model_config: Sam2Config | None = Sam2Config()):
        self.model_config = model_config
        self.image_model = SAM2ImagePredictor.from_pretrained(
            self.model_config.hf_model
        )
        self.video_model = SAM2VideoPredictor.from_pretrained(
            self.model_config.hf_model
        )
        self._torch_dtype = (
            torch.bfloat16 if self.model_config.device == Device.GPU else torch.float16
        )
        if (
            self.model_config.device == Device.GPU
            and torch.cuda.get_device_properties(0).major >= 8
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def __call__(
        self,
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
        *,
        bboxes: list[ODResponse] | None = None,
        chunk_length_frames: int | None = 20,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
        multimask_output: bool = False,
        iou_threshold: float = 0.6,
    ) -> list[list[dict[str, Any]]]:
        """Run Sam2 on images or video and find segments based on the input.

        Returns:
            list[list[ObjMaskLabel]]:
                If bboxes are None, it returns an object with the masks, scores
                and logits. Image example:
                    [[{
                        "id": 0,
                        "mask": rle,
                        "score": 0.5,
                        "logits": HW,
                    }]]
                Video example:
                    [[{
                        "id": 0,
                        "mask": rle,
                        "score": None,
                        "logits": None,
                    }]]

            list[list[ObjBboxAndMaskLabel]:
                If bboxes are not None it includes the masks alongside the bboxes.
                and labels. For example:
                    [[{
                        "id": 0,
                        "mask": rle,
                        "label": "car",
                        "bbox": [0.1, 0.2, 0.3, 0.4]
                    }]]
        """
        Florence2Sam2Request(
            images=images,
            video=video,
            input_box=input_box,
            input_points=input_points,
            input_label=input_label,
            multimask_output=multimask_output,
        )

        predictions = []
        if images is not None:
            for idx, image in enumerate(images):
                bboxes_per_frame = None if bboxes is None else bboxes[idx]
                predictions.append(
                    self._predict_image(
                        image,
                        bboxes_per_frame=bboxes_per_frame,
                        input_box=input_box,
                        input_points=input_points,
                        input_label=input_label,
                        multimask_output=multimask_output,
                    )
                )
        elif video is not None:
            if bboxes is not None:
                predictions = self._predict_video_with_bboxes(
                    video,
                    bboxes,
                    chunk_length_frames=chunk_length_frames,
                    iou_threshold=iou_threshold,
                )
            else:
                predictions = self._predict_video(
                    video,
                    input_box=input_box,
                    input_points=input_points,
                    input_label=input_label,
                )

        return _serialize(predictions)

    def to(self, device: Device):
        raise NotImplementedError("The method 'to' is not implemented.")

    @torch.inference_mode()
    def _predict_image_model(
        self,
        image: Image.Image,
        *,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
        multimask_output: bool = False,
    ) -> Sam2BitMask:
        self.image_model.reset_predictor()
        with torch.autocast(
            device_type=self.model_config.device, dtype=self._torch_dtype
        ):
            self.image_model.set_image(np.array(image, dtype=np.uint8))
            masks, scores, logits = self.image_model.predict(
                point_coords=input_points,
                point_labels=input_label,
                box=input_box,
                multimask_output=multimask_output,
            )

        return Sam2BitMask(
            masks=masks,
            scores=scores,
            logits=logits,
        )

    def _predict_image(
        self,
        image: Image.Image,
        *,
        bboxes_per_frame: ODResponse | None = None,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
        multimask_output: bool = False,
    ) -> list[ObjMaskLabel] | list[ObjBboxAndMaskLabel]:
        """Process the input image with the Sam2 image predictor using the given prompts.

        Args:
            image:
                Input image to be processed.
            bboxes_per_frame:
                Bboxes predictions for the image.
            input_box:
                Coordinates for boxes.d
            input_points:
                Coordinates for points.
            input_label:
                Labels for the points.
            multimask_output:
                Flag whether to output multiple masks.

        Returns:
            list[ObjMaskLabel] | list[ObjBboxAndMaskLabel]:
                The output of the Sam2 model based on the input image.
                list[ObjBboxAndMaskLabel] if bboxes_per_frame is not None:
                [{
                    "id": 0,
                    "mask": rle,
                    "label": "car",
                    "bbox": [0.1, 0.2, 0.3, 0.4]
                }]
        """
        if bboxes_per_frame is not None:
            return self._get_bbox_and_mask_objs(image, bboxes_per_frame)

        return self._get_mask_objs(
            image,
            input_box=input_box,
            input_points=input_points,
            input_label=input_label,
            multimask_output=multimask_output,
        )

    def _predict_video_with_bboxes(
        self,
        video: VideoNumpy,
        bboxes: list[ODResponse],
        *,
        chunk_length_frames: int | None = 20,
        iou_threshold: float = 0.6,
    ) -> list[list[ObjBboxAndMaskLabel]]:
        """Process the input video with the SAM2 video predictor using the given prompts.

        Returns:
            list[list[ObjBboxAndMaskLabel]]:
                The output of the Sam2 model based on the input image.
                [[{
                    "id": 0,
                    "mask": rle,
                    "label": "car",
                    "bbox": [0.1, 0.2, 0.3, 0.4]
                }]]
        """
        video_shape = video.shape
        num_frames = video_shape[0]
        video_segments = []
        objects_count = 0
        last_chunk_frame_pred: list[ObjBboxAndMaskLabel] = []

        if chunk_length_frames is None:
            chunk_length_frames = num_frames

        with torch.autocast(
            device_type=self.model_config.device, dtype=self._torch_dtype
        ):
            if self.model_config.device is Device.CPU:
                inference_state = self.video_model.init_state(
                    video=video, offload_video_to_cpu=True, offload_state_to_cpu=True
                )
            else:
                inference_state = self.video_model.init_state(video=video)

            # Process each chunk in the video
            for start_frame_idx in range(0, num_frames, chunk_length_frames):
                self.video_model.reset_state(inference_state)

                next_frame_idx = (
                    start_frame_idx + chunk_length_frames
                    if (start_frame_idx + chunk_length_frames) < num_frames
                    else num_frames - 1
                )

                if (
                    bboxes[start_frame_idx] is None
                    or len(bboxes[start_frame_idx].bboxes) == 0
                ):
                    _LOGGER.debug("Skipping predictions due to empty bounding boxes")

                    num_frames = next_frame_idx - start_frame_idx
                    video_segments.extend([[] for _ in range(num_frames)])
                    continue

                objs = self._predict_image(
                    Image.fromarray(video[start_frame_idx]),
                    bboxes_per_frame=bboxes[start_frame_idx],
                )

                # updates the predictions based on the predictions overlaps
                updated_objs, objects_count = _update_reference_predictions(
                    last_chunk_frame_pred,
                    objs,
                    objects_count,
                    iou_threshold,
                )

                # Add new label points to the video predictor coming from the
                # bboxes predictions
                annotation_id_to_label = {}
                for updated_obj in updated_objs:
                    annotation_id = updated_obj.id
                    annotation_id_to_label[annotation_id] = updated_obj.label
                    _, _, out_mask_logits = self.video_model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=start_frame_idx,
                        obj_id=annotation_id,
                        box=updated_obj.bbox,
                    )

                # Propagate the predictions on the given video segment (chunk)
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.video_model.propagate_in_video(
                    inference_state, start_frame_idx, chunk_length_frames
                ):
                    if out_frame_idx not in range(len(video_segments)):
                        video_segments.append([])

                    for i, out_obj_id in enumerate(out_obj_ids):
                        pred_mask = (out_mask_logits[i][0] > 0.0).cpu().numpy()
                        if np.max(pred_mask) == 0:
                            continue

                        bbox = _mask_to_bbox(pred_mask)
                        video_segments[out_frame_idx].append(
                            ObjBboxAndMaskLabel(
                                label=annotation_id_to_label[out_obj_id],
                                bbox=bbox,
                                mask=pred_mask,
                                id=out_obj_id,
                            )
                        )

                # Save the last frame predictions to later update the newly found
                # object ids
                last_chunk_frame_pred = video_segments[next_frame_idx]

        return video_segments

    def _predict_video(
        self,
        video: VideoNumpy,
        *,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
    ) -> list[list[ObjMaskLabel]]:
        """Process the input video with the SAM2 video predictor using the given prompts.

        Returns:
            list[list[ObjMaskLabel]]:
                The output of the Sam2 model based on the input image.
                [[{
                    "id": 0,
                    "mask": rle,
                    "scores": None,
                    "logits": None
                }]]
        """
        video_shape = video.shape
        num_frames = video_shape[0]
        video_segments = []

        with torch.autocast(
            device_type=self.model_config.device, dtype=self._torch_dtype
        ):
            if self.model_config.device is Device.CPU:
                inference_state = self.video_model.init_state(
                    video=video, offload_video_to_cpu=True, offload_state_to_cpu=True
                )
            else:
                inference_state = self.video_model.init_state(video=video)

            # Process each frame in the video
            for frame_idx in range(num_frames):
                self.video_model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                    box=input_box,
                    points=input_points,
                    labels=input_label,
                )

            # Propagate the masklets across the video
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_model.propagate_in_video(inference_state):
                if out_frame_idx not in range(len(video_segments)):
                    video_segments.append([])

                for i, out_obj_id in enumerate(out_obj_ids):
                    pred_mask = (out_mask_logits[i][0] > 0.0).cpu().numpy()
                    if np.max(pred_mask) == 0:
                        continue

                    video_segments[out_frame_idx].append(
                        ObjMaskLabel(
                            score=None, logits=None, mask=pred_mask, id=out_obj_id
                        )
                    )

            self.video_model.reset_state(inference_state)

        return video_segments

    def _get_mask_objs(
        self,
        image: Image.Image,
        *,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
        multimask_output: bool = False,
    ) -> list[ObjMaskLabel]:
        annotations = []
        sam2_image_pred = self._predict_image_model(
            image,
            input_box=input_box,
            input_points=input_points,
            input_label=input_label,
            multimask_output=multimask_output,
        )

        for idx in range(len(sam2_image_pred.masks)):
            sam2_mask = sam2_image_pred.masks[idx]
            sam2_logits = sam2_image_pred.logits[idx]
            mask = sam2_mask[0, :, :] if len(sam2_mask.shape) == 3 else sam2_mask[:, :]
            logits = (
                sam2_logits[0, :, :]
                if len(sam2_logits.shape) == 3
                else sam2_logits[:, :]
            )
            annotations.append(
                ObjMaskLabel(
                    id=idx,
                    score=sam2_image_pred.scores[idx],
                    mask=mask,
                    logits=logits,
                )
            )
        return annotations

    def _get_bbox_and_mask_objs(
        self, image: Image.Image, bboxes_per_frame: ODResponse | None = None
    ) -> list[ObjBboxAndMaskLabel]:
        annotations = []
        preds = [
            {
                "bbox": bboxes_per_frame.bboxes[idx],
                "label": bboxes_per_frame.labels[idx],
            }
            for idx in range(len(bboxes_per_frame.labels))
        ]

        # there was no bbox
        if len(preds) == 0:
            return annotations

        sam2_image_pred = self._predict_image_model(
            image,
            input_box=[elt["bbox"] for elt in preds],
            input_points=None,
            input_label=None,
            multimask_output=False,
        )

        for idx in range(len(preds)):
            sam2_mask = sam2_image_pred.masks[idx]
            mask = sam2_mask[0, :, :] if len(sam2_mask.shape) == 3 else sam2_mask[:, :]
            annotations.append(
                ObjBboxAndMaskLabel(
                    id=idx,
                    bbox=preds[idx]["bbox"],
                    mask=mask,
                    label=preds[idx]["label"],
                )
            )
        return annotations


def _update_reference_predictions(
    last_predictions: list[ObjBboxAndMaskLabel],
    new_predictions: list[ObjBboxAndMaskLabel],
    objects_count: int,
    iou_threshold: float = 0.8,
) -> tuple[list[ObjBboxAndMaskLabel], int]:
    """Updates the predictions based on the predictions overlaps.

    Parameters:
        last_predictions:
            List containing the annotation predictions of the last frame's prediction
            of the video propagation.
        new_predictions:
            List containing the annotation predictions of the FlorenceV2 model prediction.
        iou_threshold:
            The IoU threshold value used to compare last_predictions and new_predictions
            annotation objects.

    Returns:
        tuple[list[ObjBboxAndMaskLabel], int]:
            The first element of the tuple is the updated list of annotations,
            the second element is the updated object count.
    """
    new_annotation_predictions: list[ObjBboxAndMaskLabel] = []
    for idx, new_prediction in enumerate(new_predictions):
        new_ann_id: int = 0
        for last_prediction in last_predictions:
            iou = calculate_mask_iou(
                new_prediction.mask,
                last_prediction.mask,
            )
            if iou > iou_threshold:
                new_ann_id = last_prediction.id
                break

        if not new_ann_id:
            objects_count += 1
            new_ann_id = objects_count
            new_predictions[idx].id = new_ann_id
            new_annotation_predictions.append(new_predictions[idx])

    updated_predictions = last_predictions + new_annotation_predictions
    return (updated_predictions, objects_count)


def _mask_to_bbox(mask: np.ndarray) -> list[int]:
    rows, cols = np.where(mask)
    if len(rows) > 0 and len(cols) > 0:
        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)
        return [x_min, y_min, x_max, y_max]


def _serialize(
    frames: list[list[ObjMaskLabel]] | list[list[ObjBboxAndMaskLabel]],
) -> list[list[dict[str, Any]]]:
    return [
        [detection.model_dump() for detection in detections] for detections in frames
    ]
