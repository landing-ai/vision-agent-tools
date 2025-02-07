from typing import List, Dict, Any
import torch
from PIL import Image
from pydantic import ConfigDict, validate_arguments
from transformers import AutoProcessor, AutoModel
from transformers import SiglipProcessor, SiglipModel

from vision_agent_tools.shared_types import BaseMLModel, Device


class Siglip(BaseMLModel):
    """
    Tool for object detection using the pre-trained Siglip model.
    This tool takes a prompt as input and generates an image using the Siglip model.
    """

    config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model_name: str | None = "google/siglip-so400m-patch14-384",
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
        labels: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Performs image classification using the Siglip model and candidate labels.

        Args:
            - image (Image.Image): The image to classify.
            - labels (List[str]): The list of candidate labels to classify the image.

        Returns:
            Dict[str, List[Any]]: The classification results, containing the labels list and scores list.
        """

        texts = [f"This is a photo of {label}." for label in labels]

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

        results = {"scores": [], "labels": []}
        for i, label in enumerate(labels):
            results["labels"].append(label)
            results["scores"].append(round(probs[0, i].item(), 4))

        return results

    def to(self, device: Device):
        self._model.to(device)
        self._processor.to(device)
        self.device = device
