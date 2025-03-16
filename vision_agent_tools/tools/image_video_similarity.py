from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import validate_call
from transformers import AutoModel, AutoProcessor

from vision_agent_tools.tools.shared_types import BaseTool
from vision_agent_tools.types import VideoNumpy


_HF_CLIP_MODEL = "openai/clip-vit-large-patch14"
_HF_SIGLIP_MODEL = "google/siglip-so400m-patch14-384"


class ImageVideoSimilarity(BaseTool):
    """
    Takes in a video and a target image and caculates the similarity between the target
    image and each frame of the text.
    """
    def __init__(self, device: str = "cuda", model: str = "siglip"):
        """
        Initializes the ImageVideoSimilarity object.
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
        target_image: Image.Image,
    ) -> list[float]:
        """
        Takes a video and target image and returns the similarity scores between each
        frame and the target image.

        Args:
            video (VideoNumpy): The input video to be processed.
            target_image: (Image): The target image image to used to calculate the
                similarity.
        """

        target_image = target_image.convert("RGB")
        inputs = self.processor(images=target_image, return_tensors="pt")
        with torch.autocast(self.device):
            inputs.to(self.device)
            outputs = self.model.get_image_features(**inputs)


        target = outputs.detach()
        frame_embs = []
        for frame in video:
            inputs = self.processor(images=frame, return_tensors="pt")
            with torch.autocast(self.device):
                inputs.to(self.device)
                outputs = self.model.get_image_features(**inputs)
            frame_embs.append(outputs.squeeze().detach())
        frame_embs = torch.stack(frame_embs)

        sims = F.cosine_similarity(target, frame_embs.unsqueeze(1), dim=-1).squeeze()
        return sims.detach().cpu().numpy().tolist()
