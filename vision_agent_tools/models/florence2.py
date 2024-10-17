import logging
from typing import Optional, Any

import torch
import numpy as np
from PIL import Image
from typing_extensions import Self
from transformers import AutoModelForCausalLM, AutoProcessor
from pydantic import Field, BaseModel, model_validator, ConfigDict

from vision_agent_tools.shared_types import (
    VideoNumpy,
    PromptTask,
    Florence2ModelName,
    Device,
    Florence2ResponseType,
    BaseMLModel,
    ODResponse,
    Florence2OCRResponse,
    Florence2TextResponse,
    Florence2OpenVocabularyResponse,
    Florence2SegmentationResponse,
)
from vision_agent_tools.models.utils import calculate_bbox_iou

_MODEL_REVISION_PER_MODEL_NAME = {
    Florence2ModelName.FLORENCE_2_BASE_FT: "refs/pr/20",
    Florence2ModelName.FLORENCE_2_LARGE: "refs/pr/65",
}
_LOGGER = logging.getLogger(__name__)
_AREA_THRESHOLD = 0.82


class Florence2Request(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: PromptTask = Field(description="The task to be performed on the image/video.")
    prompt: str | None = Field(
        "", description="The text input that complements the prompt task."
    )
    images: list[Image.Image] | None = None
    video: VideoNumpy | None = None
    batch_size: int = Field(
        5,
        ge=1,
        description="The batch size used for processing multiple images or video frames.",
    )
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


class Florence2(BaseMLModel):
    """Florence2 model.
    It supported both zero-shot and fine-tuned settings.
    For the zero-shot we use the [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large).
    For fine-tuning we use the [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft).
    This model can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation.
    """

    def __init__(
        self, model_name: Florence2ModelName, *, device: Device | None = Device.GPU
    ):
        """Initializes the Florence2 model."""
        self._base_model_name = model_name
        self._device = device
        self._fine_tuned = False
        self._fine_tune_supported_tasks = [
            PromptTask.CAPTION_TO_PHRASE_GROUNDING,
            PromptTask.OBJECT_DETECTION,
            PromptTask.CAPTION,
            PromptTask.OCR_WITH_REGION,
        ]
        self.load_base()

    def load_base(self) -> None:
        """Load the base Florence-2 model."""
        self.load(
            self._base_model_name.value,
            self._base_model_name.value,
            revision=_MODEL_REVISION_PER_MODEL_NAME[self._base_model_name.value],
        )
        self._fine_tuned = False

    def fine_tune(self, checkpoint: str) -> None:
        """Load the fine-tuned Florence-2 model."""
        _LOGGER.info("Fine-tuning the Florence-2 model.")
        self.load(checkpoint, checkpoint)
        self._fine_tuned = True

    @torch.inference_mode()
    def __call__(
        self,
        task: PromptTask,
        prompt: Optional[str] = "",
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
        *,
        batch_size: int = 5,
        nms_threshold: float = 1.0,
    ) -> Florence2ResponseType:
        """
        Performs inference on the Florence-2 model based on the provided task,
        images or video, and prompt.

        Args:
            task:
                The task to be performed on the images or video.
            prompt:
                The text input that complements the prompt task.
            images:
                A list of images for the model to process. None if using video.
            video:
                A NumPy representation of the video for inference. None if using images.
            batch_size:
                The batch size used for processing multiple images or video frames.
            nms_threshold:
                The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).

        Returns:
            Florence2ResponseType:
                The output of the Florence-2 model based on the task and prompt.
        """
        Florence2Request(
            task=task,
            prompt=prompt,
            images=images,
            video=video,
            batch_size=batch_size,
            nms_threshold=nms_threshold,
        )

        if self._fine_tuned and task not in self._fine_tune_supported_tasks:
            raise ValueError(
                f"The task {task.value} is not supported yet if your are using a fine-tuned model ."
            )

        if prompt is None:
            text_input = task
        else:
            text_input = task + prompt

        if video is not None:
            # original shape: BHWC. When using florence2 with numpy, it expects the
            # shape BCHW, where B is the amount of frames, C is a number of channels,
            # H and W are image height and width.
            images = np.transpose(video, (0, 3, 1, 2))

        parsed_answers = []
        for idx in range(0, len(images), batch_size):
            end_frame = idx + batch_size
            if end_frame >= len(images):
                end_frame = len(images)

            _LOGGER.info(
                f"Processing chunk, start frame: {idx}, end frame: {end_frame - 1}"
            )
            images_chunk = images[idx:end_frame]
            text_input_chunk = [text_input] * len(images_chunk)
            inputs = self._processor(
                text=text_input_chunk, images=images_chunk, return_tensors="pt"
            ).to(self._device.value)

            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=False,
                do_sample=False,
            )

            skip_special_tokens = False
            if task is PromptTask.OCR:
                skip_special_tokens = True

            generated_texts = self._processor.batch_decode(
                generated_ids, skip_special_tokens=skip_special_tokens
            )

            for generated_text, image in zip(generated_texts, images_chunk):
                if isinstance(image, np.ndarray):
                    image_size = image.shape[1:]
                else:
                    image_size = image.size

                parsed_answer = self._processor.post_process_generation(
                    generated_text, task=task, image_size=image_size
                )
                if (
                    task == PromptTask.CAPTION_TO_PHRASE_GROUNDING
                    or task == PromptTask.OBJECT_DETECTION
                    or task == PromptTask.DENSE_REGION_CAPTION
                    or task == PromptTask.OPEN_VOCABULARY_DETECTION
                    or task == PromptTask.REGION_PROPOSAL
                ):
                    parsed_answer[task] = _filter_predictions(
                        parsed_answer[task], image_size, nms_threshold
                    )

                parsed_answers.append(parsed_answer)

        # serialize the florence2 results
        detections = []
        for parsed_answer in parsed_answers:
            detection = parsed_answer[task]
            match task:
                case (
                    PromptTask.CAPTION_TO_PHRASE_GROUNDING
                    | PromptTask.OBJECT_DETECTION
                    | PromptTask.DENSE_REGION_CAPTION
                    | PromptTask.REGION_PROPOSAL
                ):
                    detections.append(
                        ODResponse(
                            bboxes=detection["bboxes"], labels=detection["labels"]
                        )
                    )
                case PromptTask.OCR_WITH_REGION:
                    detections.append(
                        Florence2OCRResponse(
                            quad_boxes=detection["quad_boxes"],
                            labels=detection["labels"],
                        )
                    )
                case (
                    PromptTask.CAPTION
                    | PromptTask.OCR
                    | PromptTask.DETAILED_CAPTION
                    | PromptTask.MORE_DETAILED_CAPTION
                    | PromptTask.REGION_TO_CATEGORY
                    | PromptTask.REGION_TO_DESCRIPTION
                ):
                    detections.append(Florence2TextResponse(text=detection))
                case PromptTask.OPEN_VOCABULARY_DETECTION:
                    detections.append(
                        Florence2OpenVocabularyResponse(
                            bboxes=detection["bboxes"],
                            bboxes_labels=detection["bboxes_labels"],
                            polygons=detection["polygons"],
                            polygons_labels=detection["polygons_labels"],
                        )
                    )
                case (
                    PromptTask.REFERRING_EXPRESSION_SEGMENTATION
                    | PromptTask.REGION_TO_SEGMENTATION
                ):
                    detections.append(
                        Florence2SegmentationResponse(
                            polygons=detection["polygons"],
                            labels=detection["labels"],
                        )
                    )
                case _:
                    raise ValueError(f"Task {task} not supported")

        return _serialize(detections)

    def load(
        self, model_name: str, processor_name: str, revision: str | None = None
    ) -> None:
        _LOGGER.info(f"Loading: {model_name=}, {processor_name=}, {revision=}")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, revision=revision
        )
        self._processor = AutoProcessor.from_pretrained(
            processor_name, trust_remote_code=True, revision=revision
        )
        self._model.to(self._device.value)
        self._model.eval()
        _LOGGER.info(f"Model loaded: {model_name=}, {processor_name=}, {revision=}")

    def to(self, device: Device) -> None:
        raise NotImplementedError("This method is not supported for Florence2 model.")


def _filter_predictions(
    predictions: dict[str, Any],
    image_size: tuple[int, int],
    nms_threshold: float,
) -> dict[str, Any]:
    new_preds = {}

    # Remove the whole image bounding box if it is predicted
    bboxes_to_remove = _remove_whole_image_bbox(predictions, image_size)
    new_preds = _remove_bboxes(predictions, bboxes_to_remove)

    # Apply a dummy agnostic Non-Maximum Suppression (NMS) to get rid of any
    # overlapping predictions on the same object
    bboxes_to_remove = _dummy_agnostic_nms(new_preds, nms_threshold)
    new_preds = _remove_bboxes(new_preds, bboxes_to_remove)

    return new_preds


def _remove_whole_image_bbox(
    predictions: dict[str, Any], image_size: tuple[int, int]
) -> list[int]:
    # TODO: remove polygons that covers the whole image
    bboxes_to_remove = []
    img_area = image_size[0] * image_size[1]
    for idx, bbox in enumerate(predictions["bboxes"]):
        x1, y1, x2, y2 = bbox
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / img_area > _AREA_THRESHOLD:
            _LOGGER.warning(
                "Model predicted the whole image bounding box, therefore we are "
                f"removing this prediction: {bbox}, image size: {image_size}."
            )
            bboxes_to_remove.append(idx)
    return bboxes_to_remove


def _remove_bboxes(
    predictions: dict[str, Any], bboxes_to_remove: list[int]
) -> dict[str, Any]:
    new_preds = {}
    for key, value in predictions.items():
        new_preds[key] = [
            value[idx] for idx in range(len(value)) if idx not in bboxes_to_remove
        ]
    return new_preds


def _dummy_agnostic_nms(predictions: dict[str, Any], nms_threshold: float) -> list[int]:
    """
    Applies a dummy agnostic Non-Maximum Suppression (NMS) to filter overlapping predictions.

    Parameters:
        predictions:
            All predictions, including bboxes and labels.
        nms_threshold:
            The IoU threshold value used for NMS.

    Returns:
        list[int]: Indexes to remove from the predictions.
    """
    bboxes_to_keep = []
    prediction_items = {idx: pred for idx, pred in enumerate(predictions["bboxes"])}

    while prediction_items:
        # the best prediction here is the first prediction since florence2 don't
        # have score per prediction
        best_prediction_idx = next(iter(prediction_items))
        best_prediction_bbox = prediction_items[best_prediction_idx]
        bboxes_to_keep.append(best_prediction_idx)

        new_prediction_items = {}
        for idx, pred in prediction_items.items():
            if calculate_bbox_iou(best_prediction_bbox, pred) < nms_threshold:
                bboxes_to_keep.append(idx)
                new_prediction_items[idx] = pred
        prediction_items = new_prediction_items

    bboxes_to_remove = []
    for idx, bbox in enumerate(predictions["bboxes"]):
        if idx not in bboxes_to_keep:
            _LOGGER.warning(
                "Model predicted overlapping bounding boxes, therefore we are "
                f"removing this prediction: {bbox}."
            )
            bboxes_to_remove.append(idx)

    return bboxes_to_remove


def _serialize(detections: Florence2ResponseType) -> list[dict[str, Any]]:
    return [
        detection.model_dump() if detection is not None else None
        for detection in detections
    ]
