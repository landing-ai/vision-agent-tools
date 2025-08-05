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
from vision_agent_tools.models.utils import get_device
from vision_agent_tools.helpers.filters import filter_bbox_predictions

_MODEL_REVISION_PER_MODEL_NAME = {
    Florence2ModelName.FLORENCE_2_BASE_FT: "refs/pr/20",
    Florence2ModelName.FLORENCE_2_LARGE: "refs/pr/65",
}
_LOGGER = logging.getLogger(__name__)


class Florence2Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: Florence2ModelName = Field(
        default=Florence2ModelName.FLORENCE_2_LARGE,
        description="Name of the model",
    )
    device: Device = Field(
        default=get_device(),
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. "
        "Default is the first available GPU.",
    )
    fine_tuned_model_path: str | None = Field(
        default=None,
        description="Path to the fine-tuned model checkpoint. If provided, the model will be fine-tuned.",
    )


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
        0.3,
        ge=0.1,
        le=1.0,
        description="The IoU threshold value used to apply a dummy agnostic Non-Maximum Suppression (NMS).",
    )
    chunk_length_frames: int | None = None

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

    def __init__(self, model_config: Florence2Config | None = Florence2Config()):
        """Initializes the Florence2 model."""
        self._model_config = model_config
        self._fine_tuned = False
        self._fine_tune_supported_tasks = [
            PromptTask.CAPTION_TO_PHRASE_GROUNDING,
            PromptTask.OBJECT_DETECTION,
            PromptTask.CAPTION,
            PromptTask.OCR_WITH_REGION,
        ]
        if self._model_config.fine_tuned_model_path is not None:
            self.fine_tune(self._model_config.fine_tuned_model_path)
        else:
            self.load_base()

    @torch.inference_mode()
    def __call__(
        self,
        task: PromptTask,
        prompt: Optional[str] = "",
        images: list[Image.Image] | None = None,
        video: VideoNumpy | None = None,
        *,
        batch_size: int = 5,
        nms_threshold: float = 0.3,
        chunk_length_frames: int | None = None,
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
            chunk_length_frames:
                The number of frames for each chunk of video to analyze.
                The last chunk may have fewer frames.

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
            chunk_length_frames=chunk_length_frames,
        )

        if self._fine_tuned and task not in self._fine_tune_supported_tasks:
            raise ValueError(
                f"The task {task.value} is not supported yet if your are using a fine-tuned model ."
            )

        if prompt is None:
            text_input = task
        else:
            text_input = task + prompt

        if images is not None:
            images = [image.convert("RGB") for image in images]

        if video is not None:
            # original shape: BHWC. When using florence2 with numpy, it expects the
            # shape BCHW, where B is the amount of frames, C is a number of channels,
            # H and W are image height and width.
            # TODO: fix predictions with numpy
            # images = np.transpose(video, (0, 3, 1, 2))
            images = [Image.fromarray(frame).convert("RGB") for frame in video]
            if chunk_length_frames is not None:
                # run only the start index for each chunk, this is useful
                # for florence2sam2 to optimize performance
                num_frames = video.shape[0]
                idxs_to_pred = list(range(0, num_frames, chunk_length_frames))
                images = [
                    (
                        Image.fromarray(frame).convert("RGB")
                        if idx in idxs_to_pred
                        else None
                    )
                    for idx, frame in enumerate(video)
                ]
                result = self._predict_all(task, text_input, images, nms_threshold)
                return _serialize(task, result)

        result = self._predict_batch(
            task, text_input, images, batch_size, nms_threshold
        )
        return _serialize(task, result)

    def load_base(self) -> None:
        """Load the base Florence-2 model."""
        _LOGGER.info("Loading base model for Florence-2 model.")
        self._load(
            self._model_config.model_name.value,
            self._model_config.model_name.value,
            revision=_MODEL_REVISION_PER_MODEL_NAME[self._model_config.model_name],
        )
        self._fine_tuned = False

    def fine_tune(self, checkpoint: str) -> None:
        """Load the fine-tuned Florence-2 model."""
        _LOGGER.info("Fine-tuning the Florence-2 model.")
        self._load(
            checkpoint,
            checkpoint,
            revision=_MODEL_REVISION_PER_MODEL_NAME[
                Florence2ModelName.FLORENCE_2_BASE_FT
            ],
        )
        self._fine_tuned = True

    def to(self, device: Device) -> None:
        raise NotImplementedError("This method is not supported for Florence2 model.")

    def _load(
        self, model_name: str, processor_name: str, revision: str | None = None
    ) -> None:
        _LOGGER.info(f"Loading: {model_name=}, {processor_name=}, {revision=}")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, revision=revision
        )
        self._processor = AutoProcessor.from_pretrained(
            processor_name, trust_remote_code=True, revision=revision
        )
        self._model.to(self._model_config.device.value)
        self._model.eval()
        _LOGGER.info(f"Model loaded: {model_name=}, {processor_name=}, {revision=}")

    def _predict_all(
        self,
        task: PromptTask,
        text_input: str,
        images: list[Image.Image],
        nms_threshold: float,
    ) -> list[dict[str, Any]]:
        parsed_answers = []
        for image in images:
            if image is not None:
                parsed_answers.append(
                    self._predict(task, text_input, image, nms_threshold)
                )
            else:
                parsed_answers.append(None)
        return parsed_answers

    def _predict_batch(
        self,
        task: PromptTask,
        text_input: str,
        images: list[Image.Image],
        batch_size: int,
        nms_threshold: float,
    ) -> list[dict[str, Any]]:
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
            parsed_answers.extend(
                self._batch_call(task, text_input_chunk, images_chunk, nms_threshold)
            )

        return parsed_answers

    def _predict(
        self,
        task: PromptTask,
        text_input: str,
        image: Image.Image,
        nms_threshold: float,
    ) -> dict[str, Any]:
        images_chunk = [image]
        text_input_chunk = [text_input]
        return self._batch_call(task, text_input_chunk, images_chunk, nms_threshold)[0]

    def _batch_call(
        self,
        task: PromptTask,
        text_input_chunk: list[str],
        images_chunk: list[Image.Image],
        nms_threshold: float,
    ) -> list[dict[str, Any]]:
        parsed_answers = []
        inputs = self._processor(
            text=text_input_chunk, images=images_chunk, return_tensors="pt"
        ).to(self._model_config.device.value)

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
                label_key = (
                    "bboxes_labels"
                    if task is PromptTask.OPEN_VOCABULARY_DETECTION
                    else "labels"
                )
                parsed_answer[task] = filter_bbox_predictions(
                    parsed_answer[task],
                    image_size,
                    nms_threshold=nms_threshold,
                    label_key=label_key,
                )

            parsed_answers.append(parsed_answer)
        return parsed_answers


def _serialize(
    task: PromptTask, parsed_answers: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    # serialize the florence2 results
    detections = []
    for parsed_answer in parsed_answers:
        if parsed_answer is None:
            detections.append(None)
            continue
        detection = parsed_answer[task]
        match task:
            case (
                PromptTask.CAPTION_TO_PHRASE_GROUNDING
                | PromptTask.OBJECT_DETECTION
                | PromptTask.DENSE_REGION_CAPTION
                | PromptTask.REGION_PROPOSAL
            ):
                detections.append(
                    ODResponse(bboxes=detection["bboxes"], labels=detection["labels"])
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

    return [
        detection.model_dump() if detection is not None else None
        for detection in detections
    ]
