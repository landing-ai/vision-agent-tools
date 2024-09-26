from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.image_transforms import center_to_corners_format
from transformers.models.owlv2.image_processing_owlv2 import box_iou
from transformers.utils import TensorType

from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy


class OWLV2Config(BaseModel):
    model_name: str = Field(
        default="google/owlv2-large-patch14-ensemble",
        description="Name of the model",
    )
    processor_name: str = Field(
        default="google/owlv2-large-patch14-ensemble",
        description="Name of the processor",
    )
    confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for model predictions",
    )
    device: Device = Field(
        default=(
            Device.GPU
            if torch.cuda.is_available()
            else Device.MPS
            if torch.backends.mps.is_available()
            else Device.CPU
        ),
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. Default is the first available GPU.",
    )
    nms_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="IoU threshold for non-maximum suppression of overlapping boxes",
    )
    max_batch_size: int = Field(
        default=3,
        ge=1,
        description="Maximum number of images to process in a single batch.",
    )


class Owlv2InferenceData(BaseModel):
    """
    Represents an inference result from the Owlv2 model.
    """

    label: str = Field(description="The predicted label for the detected object")
    score: float = Field(
        description="TThe confidence score associated with the prediction (between 0 and 1)"
    )
    bbox: list[float] = Field(
        description=" A list of four floats representing the bounding box coordinates (xmin, ymin, xmax, ymax) of the detected object in the image"
    )


class Owlv2(BaseMLModel):
    """
    Tool for object detection using the pre-trained Owlv2 model from
    [Transformers](https://github.com/huggingface/transformers).

    This tool takes an image and a list of prompts as input, performs object detection using the Owlv2 model,
    and returns a list of `Owlv2InferenceData` objects containing the predicted labels, confidence scores,
    and bounding boxes for detected objects with confidence exceeding a threshold.
    """

    def __run_inference(self, images, prompts, confidence, nms_threshold):
        # Prepare texts for each image
        texts = [prompts for _ in images]  # List of lists of prompts

        # Run model inference here
        inputs = self._processor(text=texts, images=images, return_tensors="pt").to(
            self.model_config.device
        )

        # Forward pass
        with torch.autocast(device_type=self.model_config.device.value):
            outputs = self._model(**inputs)

        # Prepare target_sizes
        target_sizes = torch.tensor(
            [img.size[::-1] for img in images], device=self.model_config.device
        )

        # Convert outputs (bounding boxes and class logits) to the final predictions type
        results = self._processor.post_process_object_detection_with_nms(
            outputs=outputs,
            threshold=confidence,
            nms_threshold=nms_threshold,
            target_sizes=target_sizes,
        )

        inferences_batch = []

        for i, result in enumerate(results):
            boxes, scores, labels = (
                result["boxes"],
                result["scores"],
                result["labels"],
            )

            inferences = []
            for box, score, label in zip(boxes, scores, labels):
                box = [round(b.item(), 2) for b in box]
                inferences.append(
                    Owlv2InferenceData(
                        label=prompts[label.item()], score=score.item(), bbox=box
                    )
                )
            inferences_batch.append(inferences)

        return inferences_batch

    def __init__(self, model_config: Optional[OWLV2Config] = None):
        """
        Initializes the Owlv2 object detection tool.

        Loads the pre-trained Owlv2 processor and model from Transformers.
        """
        self.model_config = model_config or OWLV2Config()
        self._model = Owlv2ForObjectDetection.from_pretrained(
            self.model_config.model_name
        )
        self._processor = Owlv2ProcessorWithNMS.from_pretrained(
            self.model_config.processor_name
        )
        self._model.to(self.model_config.device)
        self._model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        image: Image.Image | None = None,
        video: VideoNumpy[np.uint8] | None = None,
        batch_size: int = 3,
    ) -> list[list[Owlv2InferenceData]]:
        """
        Performs object detection on an image using the Owlv2 model.

        Args:
            image (Image.Image): The input image for object detection.
            prompts (list[str]): A list of prompts to be used during inference.
                                  Currently, only one prompt is supported (list length of 1).
            video (Optional[VideoNumpy]: The input video for object detection.
            batch_size (Optional[int]): Number of frames to process in a single batch. Defaults to model's max_batch_size.

        Returns:
            Optional[list[Owlv2InferenceData]]: A list of `Owlv2InferenceData` objects containing the predicted
                                               labels, confidence scores, and bounding boxes for detected objects
                                               with confidence exceeding the threshold. Returns None if no objects
                                               are detected above the confidence threshold.
        """
        if batch_size is None:
            batch_size = self.model_config.max_batch_size

        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            image = image.convert("RGB")
            images = [image]
            inferences = self.__run_inference(
                images=images,
                prompts=prompts,
                confidence=self.model_config.confidence,
                nms_threshold=self.model_config.nms_threshold,
            )
            return inferences  # Return the inference data for the single image
        if video is not None:
            images = [Image.fromarray(frame).convert("RGB") for frame in video]
            inferences = []

            # Split images into batches
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_inferences = self.__run_inference(
                    images=batch_images,
                    prompts=prompts,
                    confidence=self.model_config.confidence,
                    nms_threshold=self.model_config.nms_threshold,
                )
                inferences.extend(batch_inferences)

            return inferences  # Return a list of inference data for each frame

    def to(self, device: Device):
        self._model.to(device=device.value)


class Owlv2ProcessorWithNMS(Owlv2Processor):
    def post_process_object_detection_with_nms(
        self,
        outputs,
        threshold: float = 0.1,
        nms_threshold: float = 0.3,
        target_sizes: Union[TensorType, List[Tuple]] = None,
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            nms_threshold (`float`, *optional*):
                IoU threshold to filter overlapping objects the raw detections.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes]).to(boxes.device)
                img_w = torch.Tensor([i[1] for i in target_sizes]).to(boxes.device)
            else:
                img_h, img_w = target_sizes.unbind(1)

        # Get the max logits and corresponding labels
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert boxes from center format to corner format ([x0, y0, x1, y1])
        boxes = center_to_corners_format(boxes)

        # Apply non-maximum suppression (NMS)
        if nms_threshold < 1.0:
            for idx in range(boxes.shape[0]):
                for i in torch.argsort(-scores[idx]):
                    if not scores[idx][i]:
                        continue
                    ious = box_iou(boxes[idx][i, :].unsqueeze(0), boxes[idx])[0][0]
                    ious[i] = -1.0  # Mask self-IoU.
                    scores[idx][ious > nms_threshold] = 0.0

        # Scale boxes to absolute coordinates
        if target_sizes is not None:
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
                boxes.device
            )
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for score, label, box in zip(scores, labels, boxes):
            valid_mask = score > threshold
            score = score[valid_mask]
            label = label[valid_mask]
            box = box[valid_mask]

            results.append({"scores": score, "labels": label, "boxes": box})

        return results
