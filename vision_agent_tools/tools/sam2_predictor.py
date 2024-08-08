from typing import Any, Dict, List, Union

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from vision_agent_tools.tools.shared_types import BaseTool
from vision_agent_tools.tools.florencev2 import Florencev2, PromptTask

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def check_valid_image(file_name: str) -> bool:
    return file_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def check_valid_video(file_name: str) -> bool:
    return file_name.endswith((".mp4", ".avi", ".mov"))


def read_frames(video_path: str) -> List[np.ndarray]:
    frames = []
    vid_cap = cv2.VideoCapture(video_path)
    success, frame = vid_cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        success, frame = vid_cap.read()
    return frames


class SAM2(BaseTool):
    def __init__(self):
        self.florence2 = Florencev2()
        sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(self.predictor)

    def get_bbox_and_mask(
        self, image: Image.Image, prompts: List[str]
    ) -> Dict[int, Dict[str, Any]]:
        objs = {0: {}}
        self.image_predictor.set_image(np.array(image))
        ann_id = 0
        for prompt in prompts:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                bboxes = self.florence2(
                    image, PromptTask.CAPTION_TO_PHRASE_GROUNDING, prompt
                )[PromptTask.CAPTION_TO_PHRASE_GROUNDING]["bboxes"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, _, _ = self.image_predictor.predict(
                    point_coords=None, point_labels=None, box=bboxes, multimask_output=False
                )
            for i in range(len(bboxes)):
                objs[0][ann_id] = {
                    "box": bboxes[i],
                    "mask": (
                        masks[i, 0, :, :] if len(masks.shape) == 4 else masks[i, :, :]
                    ),
                    "label": prompt,
                }
                ann_id += 1
        return objs

    def handle_image(
        self, media: Union[str, Image.Image], prompts: List[str]
    ) -> Dict[int, Dict[str, Any]]:
        self.image_predictor.reset_predictor()
        if isinstance(media, str) and check_valid_image(media):
            image = Image.open(media)
        else:
            image = media

        objs = self.get_bbox_and_mask(image.convert("RGB"), prompts)
        return objs

    def handle_video(self, media: str, prompts: List[str]) -> Dict[int, Dict[str, Any]]:
        frames = read_frames(media)
        # cap length at 60s of 30fps
        if len(frames) > 60 * 30:
            raise ValueError("Video too long")
        objs = self.get_bbox_and_mask(
            Image.fromarray(frames[0]).convert("RGB"), prompts
        )
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        inference_state = self.predictor.init_state(video=frames)
        for label in objs:
            for ann_id in objs[0]:
                _, _, out_mask_logits = self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=ann_id,
                    mask=objs[label][ann_id]["mask"],
                )

        obj_id_to_label = {}
        for ann_id in objs[0]:
            obj_id_to_label[ann_id] = objs[0][ann_id]["label"]

        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: {
                    "mask": (out_mask_logits[i] > 0.0).cpu().numpy(),
                    "label": obj_id_to_label[out_obj_id],
                }
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        self.predictor.reset_state(inference_state)
        return video_segments

    @torch.inference_mode()
    def __call__(self, media: Union[str, Image.Image], prompts: List[str]) -> Any:
        """Returns a dictionary where the first key is the frame index then an annotation
        ID, then a dictionary of the mask, label and possibly bbox (for images) for each
        annotation ID. For example:
        {
            0:
                {
                    0: {"mask": np.ndarray, "label": "car"},
                    1: {"mask", np.ndarray, "label": "person"}
                },
            1: ...
        }
        """
        if isinstance(media, Image.Image) or (
            isinstance(media, str) and check_valid_image(media)
        ):
            return self.handle_image(media, prompts)
        elif isinstance(media, str) and check_valid_video(media):
            return self.handle_video(media, prompts)
        else:
            raise ValueError(f"Invalid media type")
