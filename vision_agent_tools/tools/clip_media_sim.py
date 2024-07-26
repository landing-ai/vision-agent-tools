from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from vision_agent_tools.tools.shared_types import BaseTool


def check_valid_video(file_name: str) -> bool:
    return file_name.endswith((".mp4", ".avi", ".mov"))


def extract_frames(
    video: str, fps: float = 1.0, max_duration: int = 5 * 60
) -> List[Tuple[float, Image.Image]]:
    video_capture = cv2.VideoCapture(video)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames / frame_rate > max_duration:
        raise ValueError(
            f"Video length is too long, max is {max_duration} seconds got {total_frames / frame_rate}"
        )
    interval_in_frames = int(frame_rate / fps)

    frame_count = 0
    timestamp_and_frames = []
    while True:
        succcess, frame = video_capture.read()
        if not succcess:
            break

        if frame_count % interval_in_frames == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert(
                "RGB"
            )
            timestamp_and_frames.append((frame_count / frame_rate, frame_pil))
        frame_count += 1

    return timestamp_and_frames


class CLIPMediaSim(BaseTool):
    HF_MODEL = "openai/clip-vit-large-patch14"

    def __init__(self, device: str = "cuda"):
        self.model = (
            CLIPModel.from_pretrained(self.HF_MODEL, trust_remote_code=True)
            .eval()
            .to(device)
        )
        self.processor = CLIPProcessor.from_pretrained(self.HF_MODEL)
        self.device = device

    def __call__(
        self,
        video: str,
        target_image: Optional[Union[str, Image.Image]] = None,
        target_text: Optional[str] = None,
        fps: float = 2.0,
        thresh: float = 0.3,
    ) -> List[Tuple[float, float]]:
        if fps > 10 or fps < 0:
            raise ValueError(f"FPS must be in (0, 20), got {fps}")
        if thresh < 0 or thresh > 1:
            raise ValueError(f"thresh must be in (0, 1), got {thresh}")
        if not check_valid_video(video):
            raise ValueError("Invalid video file provided.")
        if (target_image is None and target_text is None) or (
            target_image is not None and target_text is not None
        ):
            raise ValueError("Must provide one of target_image or target text")
        timestamp_and_frames = extract_frames(video, fps=fps)

        if target_image is not None:
            if isinstance(target_image, str):
                target_image_pil = Image.open(target_image)
            else:
                target_image_pil = target_image
            target_image_pil = target_image_pil.convert("RGB")
            inputs = self.processor(
                images=target_image_pil,
                return_tensors="pt",
            )
            with torch.no_grad(), torch.autocast(self.device):
                inputs.to(self.device)
                outputs = self.model.get_image_features(**inputs)
        else:
            inputs = self.processor(
                text=[target_text], return_tensors="pt", padding=True
            )
            with torch.no_grad(), torch.autocast(self.device):
                inputs.to(self.device)
                outputs = self.model.get_text_features(**inputs)

        target = outputs.detach()

        frame_embs = []
        for _, f in timestamp_and_frames:
            inputs = self.processor(images=f, return_tensors="pt")
            with torch.no_grad(), torch.autocast(self.device):
                inputs.to(self.device)
                outputs = self.model.get_image_features(**inputs)
            frame_embs.append(outputs.detach())
        frame_embs = torch.stack(frame_embs)
        sims = F.cosine_similarity(frame_embs, target, dim=-1).cpu().numpy()
        times = np.array([t[0] for t in timestamp_and_frames])
        output = np.concatenate([times[:, None], sims], axis=1)
        output = output[output[:, 1] > thresh]
        return output.tolist()


if __name__ == "__main__":
    model = CLIPMediaSim()
    # out = output = model("section1_chunk_24_32.mp4", target_image="logo.png")
    # out = output = model("section1.mp4", target_image="saved_frames/frame_0.jpg", thresh=0.9)
    out = output = model("section1.mp4", target_text="soccer", thresh=0.2)
    # out = output = model("liuxiang.mp4", target_image="logo.png")
    print(out)
