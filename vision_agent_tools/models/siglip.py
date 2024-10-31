from enum import Enum
from typing import List, Dict, Any
import torch
from PIL import Image
from pydantic import Field, ConfigDict, validate_arguments
from transformers import AutoProcessor, AutoModel
from transformers import SiglipProcessor, SiglipModel

from vision_agent_tools.shared_types import BaseMLModel, Device
from pydantic import BaseModel


class SiglipTask(str, Enum):
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"


class SiglipResponseItem(BaseModel):
    label: str = Field(..., description="The label of the classification result.")
    score: float = Field(
        ..., ge=0, le=1, description="The score of the classification result."
    )


class Siglip(BaseMLModel):
    """
    Tool for object detection using the pre-trained Siglip model.
    This tool takes a prompt as input and generates an image using the Siglip model.
    """

    config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model_name: str | None = "google/siglip-base-patch16-224",
        device: Device | None = None,
    ):
        """
        Initializes the Siglip image classification tool.
        """
        self._model = None
        self._processor = None
        self.device = device

        if self.device is None:
            self.device = Device.GPU if torch.cuda.is_available() else Device.CPU

        if device == Device.GPU:
            self._model = SiglipModel.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map=device,
            )
            self._processor = SiglipProcessor.from_pretrained(model_name)
        else:
            self._model = AutoModel.from_pretrained(model_name)
            self._processor = AutoProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    @validate_arguments(config=config)
    def __call__(
        self,
        image: Image.Image,
        candidate_labels: List[str],
        task: SiglipTask = SiglipTask.ZERO_SHOT_IMAGE_CLASSIFICATION,
    ) -> List[Dict[str, Any]]:
        """
        Performs image classification using the Siglip model and candidate labels.

        Args:
            - image (Image.Image): The image to classify.
            - candidate_labels (List[str]): The list of candidate labels to classify the image.
            - task (SiglipTask): The task to perform using the model:
                - zero-shot image classification - "zero-shot-image-classification".

        Returns:
            List[Dict[str, Any]]: The list of classification results, each containing a label and a score.
        """

        if task == SiglipTask.ZERO_SHOT_IMAGE_CLASSIFICATION:
            output = self._zero_shot_classification(image, candidate_labels)
        else:
            raise ValueError(
                f"Unsupported task: {task}. Supported tasks are: {', '.join([task.value for task in SiglipTask])}."
            )

        return output

    def to(self, device: Device):
        self._model.to(device)
        self._processor.to(device)
        self.device = device

    def _zero_shot_classification(
        self, image: Image.Image, candidate_labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Classifies the image using the Siglip model and candidate labels.

        Args:
            - image (Image.Image): The image to classify.
            - candidate_labels (List[str]): The list of candidate labels to classify the image.

        Returns:
            List[Dict[str, Any]]: The list of classification results, each containing a label and a score.
        """

        texts = [f"This is a photo of {label}." for label in candidate_labels]

        inputs = self._processor(
            text=texts,
            images=image,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            with torch.autocast(self.device):
                outputs = self._model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)

        results = []
        for i, label in enumerate(candidate_labels):
            result = SiglipResponseItem(label=label, score=round(probs[0, i].item(), 4))
            results.append({"label": result.label, "score": result.score})

        return results
