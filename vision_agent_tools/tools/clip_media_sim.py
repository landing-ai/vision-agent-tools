from annotated_types import Gt, Lt, Ge
from typing_extensions import Annotated, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import validate_call
from transformers import CLIPModel, CLIPProcessor

from vision_agent_tools.tools.shared_types import BaseTool
from vision_agent_tools.types import VideoNumpy


_HF_MODEL = "openai/clip-vit-large-patch14"


class CLIPMediaSim(BaseTool):
    def __init__(self, device: str = "cuda"):
        self.model = CLIPModel.from_pretrained(_HF_MODEL).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(_HF_MODEL)
        self.device = device

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        video: VideoNumpy[np.uint8],
        timestamps: Sequence[Annotated[float, Ge(0)]],
        target_image: Image.Image | None = None,
        target_text: str | None = None,
        thresh: Annotated[float, Gt(0), Lt(1)] = 0.3,
    ) -> list[tuple[float, float]]:
        if (target_image is None and target_text is None) or (
            target_image is not None and target_text is not None
        ):
            raise ValueError("Must provide one of target_image or target_text")

        if target_image is not None:
            target_image = target_image.convert("RGB")
            inputs = self.processor(images=target_image, return_tensors="pt")
            with torch.no_grad(), torch.autocast(self.device):
                inputs.to(self.device)
                outputs = self.model.get_image_features(**inputs)
        else:
            inputs = self.processor(
                text=[target_text], return_tensors="pt", padding=True
            )
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
            frame_embs.append(outputs.detach())
        frame_embs = torch.stack(frame_embs)
        sims = F.cosine_similarity(frame_embs, target, dim=-1).cpu().numpy()
        times = np.array(timestamps)
        output = np.concatenate([times[:, None], sims], axis=1)
        output = output[output[:, 1] > thresh]
        return output.tolist()
