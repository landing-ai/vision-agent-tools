import torch
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageClassification, ViTImageProcessor

from vision_agent_tools.shared_types import BaseMLModel, Device

CHECKPOINT = "Falconsai/nsfw_image_detection"


class NSFWInferenceData(BaseModel):
    """
    Represents an inference result from the NSFWClassification model.

    Attributes:
        label (str): The predicted label for the image.
        score (float): The confidence score associated with the prediction (between 0 and 1).
    """

    label: str
    score: float


class NSFWClassification(BaseMLModel):
    """
    The primary intended use of this model is for the classification of
    [NSFW (Not Safe for Work)](https://huggingface.co/Falconsai/nsfw_image_detection) images.

    """

    def __init__(self):
        """
        Initializes the NSFW (Not Safe for Work) classification tool.
        """
        self._model = AutoModelForImageClassification.from_pretrained(CHECKPOINT)
        self._processor = ViTImageProcessor.from_pretrained(CHECKPOINT)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model.to(self.device)

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
    ) -> NSFWInferenceData:
        """
        Performs the NSFW inference on an image using the NSFWClassification model.

        Args:
            image (Image.Image): The input image for object detection.

        Returns:
            NSFWInferenceData: The inference result from the NSFWClassification model.
                label (str): The label for the unsafe content detected in the image.
                score (float):The score for the unsafe content detected in the image.
        """
        image = image.convert("RGB")
        with torch.autocast(self.device):
            inputs = self._processor(
                images=image,
                return_tensors="pt",
            ).to(self.device)
            outputs = self._model(**inputs)
        logits = outputs.logits
        scores = logits.softmax(dim=1).tolist()[0]
        predicted_label = logits.argmax(-1).item()
        text = self._model.config.id2label[predicted_label]
        return NSFWInferenceData(label=text, score=scores[predicted_label])

    def to(self, device: Device):
        self._model.to(device=device.value)
