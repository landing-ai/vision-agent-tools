from typing import Optional

import torch
from PIL import Image
from pydantic import BaseModel
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from vision_agent_tools.tools.shared_types import BaseTool

MODEL_NAME = "google/owlv2-base-patch16-ensemble"
PROCESSOR_NAME = "google/owlv2-base-patch16-ensemble"
DEFAULT_CONFIDENCE = 0.2


class Owlv2InferenceData(BaseModel):
    label: str
    score: float
    bbox: list[float]


class Owlv2(BaseTool):
    def __init__(self):
        self._processor = Owlv2Processor.from_pretrained(PROCESSOR_NAME)
        self._model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME)

    def __call__(
        self,
        image: Image.Image,
        prompts: list[str],
        confidence: Optional[float] = DEFAULT_CONFIDENCE,
    ):
        texts = [prompts]
        # Run model inference here
        inputs = self._processor(text=texts, images=image, return_tensors="pt")
        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])

        # Convert outputs (bounding boxes and class logits) to the final predictions type
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=0.1, target_sizes=target_sizes
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
            if score < confidence:
                continue
            inferences.append(
                Owlv2InferenceData(
                    label=texts[i][label.item()], score=score.item(), bbox=box
                )
            )

        if len(inferences) == 0:
            return None
        return inferences
