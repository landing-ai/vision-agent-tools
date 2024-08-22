import torch

from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageClassification, ViTImageProcessor
from vision_agent_tools.shared_types import BaseTool, CachePath
from vision_agent_tools.tools.utils import manage_hf_model_cache


class NSFWInferenceData(BaseModel):
    """
    Represents an inference result from the NSFWClassification model.

    Attributes:
        label (str): The predicted label for the image.
        score (float): The confidence score associated with the prediction (between 0 and 1).
    """

    label: str
    score: float


class NSFWClassification(BaseTool):
    """
    The primary intended use of this model is for the classification of
    [NSFW (Not Safe for Work)](https://huggingface.co/Falconsai/nsfw_image_detection) images.

    """

    _MODEL_NAME = "Falconsai/nsfw_image_detection"

    def __init__(self, cache_dir: CachePath = None):
        """
        Initializes the NSFW (Not Safe for Work) classification tool.
        """
        model_snapshot = manage_hf_model_cache(self._MODEL_NAME, cache_dir)
        self._model = AutoModelForImageClassification.from_pretrained(
            model_snapshot, trust_remote_code=True, local_files_only=True
        )
        self._processor = ViTImageProcessor.from_pretrained(
            model_snapshot, trust_remote_code=True, local_files_only=True
        )

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
