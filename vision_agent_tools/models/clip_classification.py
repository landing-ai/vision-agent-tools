import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field, validate_call
from transformers import CLIPModel, CLIPProcessor
from typing import List, Optional
from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy


class CLIPConfig(BaseModel):
    model_name: str = Field(
        default="openai/clip-vit-large-patch14",
        description="Name of the model",
    )
    processor_name: str = Field(
        default="openai/clip-vit-large-patch14",
        description="Name of the processor",
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for predictions",
    )
    device: Device = Field(
        default=Device.GPU
        if torch.cuda.is_available()
        else Device.MPS
        if torch.backends.mps.is_available()
        else Device.CPU,
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. Default is the first available GPU.",
    )


class ClipClassificationData(BaseModel):
    label: str = Field(description="Predicted label for the classification")
    score: float = Field(description="Confidence score for the classification")


class CLIPClassification(BaseMLModel):
    """
    A class that performs classification using the CLIP model.
    """

    def __init__(self, model_config: Optional[CLIPConfig] = None):
        """
        Initializes the CLIP image and video classification tool.

        Loads the pre-trained CLIP processor and model from Transformers.
        """
        self.model_config = model_config or CLIPConfig()
        self._model = CLIPModel.from_pretrained(self.model_config.model_name)
        self._processor = CLIPProcessor.from_pretrained(
            self.model_config.processor_name
        )
        self._model.to(self.model_config.device)
        self._model.eval()

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        image: Optional[Image.Image] = None,
        video: Optional[VideoNumpy[np.uint8]] = None,
        prompts: List[str] = None,
    ) -> List[ClipClassificationData]:
        """
        Run classification on the image or video based on text prompts.

        Args:
            image (Image.Image): The input image for classification.
            video (np.ndarray): The input video for classification. If provided, process video frame by frame.
            prompts (List[str]): List of text prompts for classification.

        Returns:
            List[ClipClassificationData]: A list of classification results for each prompt.
        """

        if (image is None and video is None) or (
            image is not None and video is not None
        ):
            raise ValueError("Must provide one of 'image' or 'video' but not both.")

        if prompts is None or len(prompts) == 0:
            raise ValueError("Text prompts must be provided.")

        results = []
        return results
