import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_call
from transformers import AutoModel, AutoProcessor

from vision_agent_tools.tools.shared_types import BaseTool
from vision_agent_tools.types import VideoNumpy


_HF_CLIP_MODEL = "openai/clip-vit-large-patch14"
_HF_SIGLIP_MODEL = "google/siglip-so400m-patch14-384"


class TextVideoClassifier(BaseTool):
    """
    Takes in a list of texts and a video and classifies each frame in a video according
    to the given texts.
    """

    def __init__(self, device: str = "cuda", model: str = "siglip"):
        """
        Initializes the TextVideoClassifier object with a pre-trained SigLip model.
        """
        if model == "siglip":
            model_key = _HF_SIGLIP_MODEL
        elif model == "clip":
            model_key = _HF_CLIP_MODEL
        else:
            raise ValueError(f"Unknown model type, only accepts ['siglip', 'clip']")
        self.model_key = model_key
        self.model = AutoModel.from_pretrained(model_key).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(model_key)
        self.device = device

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        video: VideoNumpy[np.uint8],
        target_text: list[str],
    ) -> list[tuple[float, float]]:
        """
        Receives a video and target text and returns a probability score for each frame
        over each target_text element.

        Args:
            video (VideoNumpy: The input video to be processed.
            target_text (list[str]): The target text used to classify. 
        """
        if len(target_text) < 2:
            raise ValueError(f"Must contain at least 2 classes")

        inputs = self.processor(text=target_text, return_tensors="pt", padding=True)
        with torch.autocast(self.device):
            inputs.to(self.device)
            outputs = self.model.get_text_features(**inputs)

        target = outputs.detach()

        frame_embs = []
        for frame in video:
            inputs = self.processor(images=frame, return_tensors="pt")
            with torch.autocast(self.device):
                inputs.to(self.device)
                outputs = self.model.get_image_features(**inputs)
            frame_embs.append(outputs.squeeze().detach())
        frame_embs = torch.stack(frame_embs)

        # first dim is frame count, second dim is taret classes
        probs = (
            (
                F.cosine_similarity(target, frame_embs.unsqueeze(1), dim=-1)
                * self.model.logit_scale.exp()
                + (self.model.logit_bias if self.model_key == "siglip" else 0)
            )
            .softmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        return probs.tolist()
