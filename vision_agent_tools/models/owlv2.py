import torch
import torch.profiler
import logging
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.image_transforms import center_to_corners_format
from transformers.utils import TensorType

from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        prep_start = time.time()
        texts = [prompts] * len(images)  # List of lists of prompts

        # Run model inference here
        inputs = self._processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(self.model_config.device)
        prep_end = time.time()
        logger.info(f"Input preparation time: {prep_end - prep_start:.4f} seconds")

        logger.info(f"Input pixel_values shape: {inputs['pixel_values'].shape}")
        logger.info(f"Input input_ids shape: {inputs['input_ids'].shape}")
        logger.info(f"Input attention_mask shape: {inputs['attention_mask'].shape}")

        # Forward pass
        infer_start = time.time()
        with torch.autocast(device_type=self.model_config.device.value):
            outputs = self._model(**inputs)
        infer_end = time.time()
        logger.info(f"Model inference time: {infer_end - infer_start:.4f} seconds")

        # Prepare target_sizes
        target_sizes = torch.tensor(
            [img.size[::-1] for img in images], device=self.model_config.device
        )

        # Convert outputs (bounding boxes and class logits) to the final predictions type
        post_start = time.time()
        results = self._processor.post_process_object_detection_with_nms(
            outputs=outputs,
            threshold=confidence,
            nms_threshold=nms_threshold,
            target_sizes=target_sizes,
        )
        post_end = time.time()
        logger.info(f"Post-processing time: {post_end - post_start:.4f} seconds")

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

        total_time = post_end - prep_start
        logger.info(f"Total inference time: {total_time:.4f} seconds")

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
        logger.info(
            f"Initialized Owlv2 model '{self.model_config.model_name}' on device '{self.model_config.device.value}'."
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        image: Image.Image | None = None,
        video: VideoNumpy[np.uint8] | None = None,
        batch_size: int | None = None,
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
            logger.info(f"Processing a single image with prompts: {prompts}")
            inferences = self.__run_inference(
                images=images,
                prompts=prompts,
                confidence=self.model_config.confidence,
                nms_threshold=self.model_config.nms_threshold,
            )
            return inferences  # Return the inference data for the single image
        if video is not None:
            images = [Image.fromarray(frame).convert("RGB") for frame in video]
            total_frames = len(images)
            logger.info(
                f"Processing video with {total_frames} frames using batch size {batch_size}."
            )
            inferences = []

            start_time = time.time()

            # Split images into batches
            # with torch.profiler.profile(
            #         activities=[
            #             torch.profiler.ProfilerActivity.CPU,
            #             torch.profiler.ProfilerActivity.CUDA,
            #         ],
            #         record_shapes=True,
            #         profile_memory=True,
            #         with_stack=True,
            #     ) as prof:

            for batch_index, i in enumerate(range(0, total_frames, batch_size)):
                batch_images = images[i : i + batch_size]
                logger.debug(
                    f"Processing batch {batch_index + 1}/{(total_frames + batch_size - 1) // batch_size} with {len(batch_images)} frames."
                )
                batch_start_time = time.time()

                batch_inferences = self.__run_inference(
                    images=batch_images,
                    prompts=prompts,
                    confidence=self.model_config.confidence,
                    nms_threshold=self.model_config.nms_threshold,
                )

                end_time = time.time()
                batch_time = end_time - batch_start_time
                logger.debug(
                    f"Processed batch {batch_index + 1} in {batch_time:.2f} seconds."
                )
                inferences.extend(batch_inferences)
            # logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

            total_time = time.time() - start_time
            logger.info(
                f"Completed processing of {total_frames} frames in {total_time}s."
            )
            return inferences  # Return a list of inference data for each frame

    def to(self, device: Device):
        self._model.to(device=device.value)


class Owlv2ProcessorWithNMS(Owlv2Processor):
    def nms(self, boxes, scores, iou_threshold):
        """
        Performs Non-Maximum Suppression (NMS) on the bounding boxes.

        Args:
            boxes (Tensor[N, 4]): Bounding boxes in (x1, y1, x2, y2) format.
            scores (Tensor[N]): Scores for each bounding box.
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            keep (Tensor): Indices of bounding boxes to keep.
        """
        # Ensure boxes and scores are tensors
        boxes = boxes.to(device=boxes.device)
        scores = scores.to(device=boxes.device)

        # Compute areas of boxes
        x1 = boxes[:, 0]  # xmin
        y1 = boxes[:, 1]  # ymin
        x2 = boxes[:, 2]  # xmax
        y2 = boxes[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)

        # Sort the detections by score in descending order
        order = scores.argsort(descending=True)

        keep = []

        while order.numel() > 0:
            idx_self = order[0].item()
            keep.append(idx_self)

            if order.numel() == 1:
                break

            idx_other = order[1:]

            # Compute IoU between the highest-scoring box and the rest
            xx1 = torch.max(x1[idx_self], x1[idx_other])
            yy1 = torch.max(y1[idx_self], y1[idx_other])
            xx2 = torch.min(x2[idx_self], x2[idx_other])
            yy2 = torch.min(y2[idx_self], y2[idx_other])

            inter_w = (xx2 - xx1).clamp(min=0)
            inter_h = (yy2 - yy1).clamp(min=0)
            inter_area = inter_w * inter_h

            union_area = areas[idx_self] + areas[idx_other] - inter_area
            iou = inter_area / union_area

            # Keep boxes with IoU less than the threshold
            mask = iou <= iou_threshold
            order = order[1:][mask]

        return torch.tensor(keep, device=boxes.device)

    def post_process_object_detection_with_nms(
        self,
        outputs,
        threshold: float = 0.1,
        nms_threshold: float = 0.3,
        target_sizes: Union[TensorType, List[Tuple]] = None,
    ):
        logits, boxes = outputs.logits, outputs.pred_boxes

        # Compute probabilities and labels
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert boxes from center format to corner format
        boxes = center_to_corners_format(boxes)

        # Scale boxes to absolute coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                target_sizes = torch.tensor(target_sizes, device=boxes.device)
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)[:, None, :]
            boxes = boxes * scale_fct

        results = []
        batch_size = logits.shape[0]
        for idx in range(batch_size):
            score = scores[idx]
            label = labels[idx]
            box = boxes[idx]

            # Apply confidence threshold
            valid_mask = score > threshold
            score = score[valid_mask]
            label = label[valid_mask]
            box = box[valid_mask]

            if box.numel() == 0:
                results.append({"scores": score, "labels": label, "boxes": box})
                continue

            # Apply NMS per image
            nms_start = time.time()
            keep = self.nms(box, score, nms_threshold)
            score = score[keep]
            label = label[keep]
            box = box[keep]
            nms_end = time.time()
            logger.info(f"NMS time: {nms_end - nms_start:.4f} second")

            results.append({"scores": score, "labels": label, "boxes": box})

        return results
