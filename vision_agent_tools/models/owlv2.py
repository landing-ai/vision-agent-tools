from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.image_transforms import center_to_corners_format
from transformers.models.owlv2.image_processing_owlv2 import box_iou
from transformers.utils import TensorType

from vision_agent_tools.models.utils import filter_redundant_boxes
from vision_agent_tools.shared_types import BaseMLModel, BboxLabel, Device, VideoNumpy


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


class Owlv2(BaseMLModel):
    """
    Tool for object detection using the pre-trained Owlv2 model from
    [Transformers](https://github.com/huggingface/transformers).

    This tool takes an image and a list of prompts as input, performs object detection using the Owlv2 model,
    and returns a list of `BboxLabel` objects containing the predicted labels, confidence scores,
    and bounding boxes for detected objects with confidence exceeding a threshold.
    """

    from typing import Dict, List

    def _filter_bboxes(self, bboxlabels: List[BboxLabel]) -> List[BboxLabel]:
        """
        Filters out redundant BboxLabel objects that fully contain multiple smaller boxes of the same label.

        Parameters:
            bboxlabels (List[BboxLabel]): List of BboxLabel objects to be filtered.

        Returns:
            List[BboxLabel]: Filtered list of BboxLabel objects.
        """
        bboxes = [bl.bbox for bl in bboxlabels]
        labels = [bl.label for bl in bboxlabels]

        filtered = filter_redundant_boxes({"bboxes": bboxes, "labels": labels})
        filtered_bboxes = filtered["bboxes"]
        filtered_labels = filtered["labels"]

        filtered_pairs = list(zip(filtered_bboxes, filtered_labels))

        # preserving the original order
        output_bboxlabels = []
        for bl in bboxlabels:
            pair = (bl.bbox, bl.label)
            if pair in filtered_pairs:
                output_bboxlabels.append(bl)
                filtered_pairs.remove(pair)  # Remove to handle duplicates correctly

        return output_bboxlabels

    def __run_inference(
        self, image, texts, confidence, nms_threshold
    ) -> list[BboxLabel]:
        # Run model inference here
        inputs = self._processor(
            text=texts,
            images=image,
            return_tensors="pt",
            truncation=True,
        ).to(self.model_config.device)
        # Forward pass
        with torch.autocast(self.model_config.device):
            outputs = self._model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])

        # Convert outputs (bounding boxes and class logits) to the final predictions type
        results = self._processor.post_process_object_detection_with_nms(
            outputs=outputs,
            threshold=confidence,
            nms_threshold=nms_threshold,
            target_sizes=target_sizes,
        )
        i = 0  # given that we are predicting on only one image
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )

        inferences: list[BboxLabel] = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            inferences.append(
                BboxLabel(label=texts[i][label.item()], score=score.item(), bbox=box)
            )

        filtered_inferences = self._filter_bboxes(inferences)

        return filtered_inferences

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
    ) -> list[list[BboxLabel]]:
        """
        Performs object detection on an image using the Owlv2 model.

        Args:
            image (Image.Image): The input image for object detection.
            prompts (list[str]): A list of prompts to be used during inference.
                Currently, only one prompt is supported (list length of 1).
            video (Optional[VideoNumpy]: The input video for object detection.

        Returns:
            list[list[BboxLabel]]: A list of `BboxLabel` objects containing the predicted
                labels, confidence scores, and bounding boxes for detected objects
                with confidence exceeding the threshold. Returns None if no objects
                are detected above the confidence threshold.
        """
        texts = [prompts]

        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            image = image.convert("RGB")
            inferences = []
            inferences.append(
                self.__run_inference(
                    image=image,
                    texts=texts,
                    confidence=self.model_config.confidence,
                    nms_threshold=self.model_config.nms_threshold,
                )
            )
        if video is not None:
            inferences = []
            for frame in video:
                image = Image.fromarray(frame).convert("RGB")
                inferences.append(
                    self.__run_inference(
                        image=image,
                        texts=texts,
                        confidence=self.model_config.confidence,
                        nms_threshold=self.model_config.nms_threshold,
                    )
                )

        return inferences

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
            `List[dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
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
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            # rescale coordinates
            width_ratio = 1
            height_ratio = 1

            if img_w < img_h:
                width_ratio = img_w / img_h
            elif img_h < img_w:
                height_ratio = img_h / img_w

            img_w = img_w / width_ratio
            img_h = img_h / height_ratio

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
                boxes.device
            )
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for score_array, label_array, box_array in zip(scores, labels, boxes):
            high_score_mask = score_array > threshold
            filtered_scores = score_array[high_score_mask]
            filtered_labels = label_array[high_score_mask]
            filtered_boxes = box_array[high_score_mask]

            results.append(
                {
                    "scores": filtered_scores,
                    "labels": filtered_labels,
                    "boxes": filtered_boxes,
                }
            )

        return results
