from typing import Optional

import torch
from PIL import Image
from pydantic import BaseModel, Field
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from vision_agent_tools.shared_types import BaseMLModel, Device

MODEL_NAME = "google/owlv2-large-patch14-ensemble"
PROCESSOR_NAME = "google/owlv2-large-patch14-ensemble"
DEFAULT_CONFIDENCE = 0.1


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

    def __init__(self):
        """
        Initializes the Owlv2 object detection tool.

        Loads the pre-trained Owlv2 processor and model from Transformers.
        """
        self._processor = Owlv2Processor.from_pretrained(PROCESSOR_NAME)
        self._model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        prompts: list[str],
        confidence: Optional[float] = DEFAULT_CONFIDENCE,
    ) -> Optional[list[Owlv2InferenceData]]:
        """
        Performs object detection on an image using the Owlv2 model.

        Args:
            image (Image.Image): The input image for object detection.
            prompts (list[str]): A list of prompts to be used during inference.
                                  Currently, only one prompt is supported (list length of 1).
            confidence (Optional[float], defaults=DEFAULT_CONFIDENCE): The minimum confidence threshold for
                                                                          including a detection in the results.

        Returns:
            Optional[list[Owlv2InferenceData]]: A list of `Owlv2InferenceData` objects containing the predicted
                                               labels, confidence scores, and bounding boxes for detected objects
                                               with confidence exceeding the threshold. Returns None if no objects
                                               are detected above the confidence threshold.
        """
        image = image.convert("RGB")
        texts = [prompts]
        # Run model inference here
        inputs = self._processor(text=texts, images=image, return_tensors="pt").to(
            self.device
        )
        # Forward pass
        with torch.autocast(self.device):
            outputs = self._model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])

        # Convert outputs (bounding boxes and class logits) to the final predictions type
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=confidence, target_sizes=target_sizes
        )
        i = 0  # given that we are predicting on only one image
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )

        inferences: list[Owlv2InferenceData] = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            inferences.append(
                Owlv2InferenceData(
                    label=texts[i][label.item()], score=score.item(), bbox=box
                )
            )

        if len(inferences) == 0:
            return None
        return inferences

    def to(self, device: Device):
        self._model.to(device=device.value)
