from annotated_types import Gt, Lt
from dataclasses import dataclass
from typing_extensions import Annotated

import torch
import numpy as np
from PIL import Image
from pydantic import validate_call

from vision_agent_tools.shared_types import BaseTool, VideoNumpy, SegmentationMask
from vision_agent_tools.tools.florencev2 import Florencev2, PromptTask

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
# import hydra

# hydra.core.global_hydra.GlobalHydra.instance().clear()
# hydra.initialize_config_module(
#     "vision_agent_tools/tools/sam2_configs", version_base="1.2"
# )


_HF_MODEL = "facebook/sam2-hiera-large"


@dataclass
class ImageBboxAndMaskLabel:
    label: str
    bounding_box: list[
        Annotated[float, "x_min", Gt(0), Lt(1)],
        Annotated[float, "y_min", Gt(0), Lt(1)],
        Annotated[float, "x_max", Gt(0), Lt(1)],
        Annotated[float, "y_max", Gt(0), Lt(1)],
    ]
    mask: SegmentationMask | None


@dataclass
class MaskLabel:
    label: str
    mask: SegmentationMask


class Florence2SAM2(BaseTool):
    def __init__(self):
        self.florence2 = Florencev2()
        self.video_predictor = SAM2VideoPredictor.from_pretrained(_HF_MODEL)
        self.image_predictor = SAM2ImagePredictor(self.video_predictor)

    @torch.inference_mode()
    def get_bbox_and_mask(
        self, image: Image.Image, prompts: list[str], return_mask: bool = True
    ) -> dict[int, ImageBboxAndMaskLabel]:
        objs = {}
        self.image_predictor.set_image(np.array(image))
        annotation_id = 0
        for prompt in prompts:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                bboxes = self.florence2(
                    image, PromptTask.CAPTION_TO_PHRASE_GROUNDING, prompt
                )[PromptTask.CAPTION_TO_PHRASE_GROUNDING]["bboxes"]
            if return_mask:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks, _, _ = self.image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bboxes,
                        multimask_output=False,
                    )
            for i in range(len(bboxes)):
                objs[annotation_id] = ImageBboxAndMaskLabel(
                    bounding_box=bboxes[i],
                    mask=(
                        masks[i, 0, :, :] if len(masks.shape) == 4 else masks[i, :, :]
                    )
                    if return_mask
                    else None,
                    label=prompt,
                )
                annotation_id += 1
        return objs

    @torch.inference_mode()
    def handle_image(
        self, image: Image.Image, prompts: list[str]
    ) -> dict[int, dict[int, ImageBboxAndMaskLabel]]:
        self.image_predictor.reset_predictor()
        objs = self.get_bbox_and_mask(image.convert("RGB"), prompts)
        return {"0": objs}

    def handle_video(
        self, video: VideoNumpy, prompts: list[str]
    ) -> dict[int, dict[int, MaskLabel]]:
        objs = self.get_bbox_and_mask(
            Image.fromarray(video[0]).convert("RGB"), prompts, return_mask=False
        )
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        inference_state = self.video_predictor.init_state(video=video)
        for annotation_id in objs:
            _, _, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=annotation_id,
                box=objs[annotation_id].bounding_box,
            )

        annotation_id_to_label = {}
        for annotation_id in objs:
            annotation_id_to_label[annotation_id] = objs[annotation_id].label

        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: MaskLabel(
                    mask=(out_mask_logits[i] > 0.0).cpu().numpy(),
                    label=annotation_id_to_label[out_obj_id],
                )
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        self.video_predictor.reset_state(inference_state)
        return video_segments

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self, media: Image.Image | VideoNumpy, prompts: list[str]
    ) -> dict[int, dict[int, ImageBboxAndMaskLabel | MaskLabel]]:
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
        if isinstance(media, Image.Image):
            return self.handle_image(media, prompts)
        elif isinstance(media, np.ndarray):
            return self.handle_video(media, prompts)
        # No need to raise an error here, the validatie_call decorator will take care of it


if __name__ == "__main__":
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import cv2

    def read_frames(video: str):
        cap = cv2.VideoCapture(video)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    model = Florence2SAM2()
    video_path = (
        "/home/camilo-zapata/landing_ai/vision-agent-tools/sample_data/football.mp4"
    )
    video = read_frames(video_path)
    prompts = ["shoe", "hand"]
    output = model(video, prompts)

    def get_text_coords_from_mask(mask, v_gap=10, h_gap=10):
        mask = mask.astype(np.uint8)
        if np.sum(mask) == 0:
            return 0, 0

        rows, cols = np.nonzero(mask)
        top = rows.min()
        bottom = rows.max()
        left = cols.min()
        right = cols.max()

        if top - v_gap < 0:
            if bottom + v_gap > mask.shape[0]:
                top = top
            else:
                top = bottom + v_gap
        else:
            top = top - v_gap

        return left + (right - left) // 2 - h_gap, top

    def masks_to_video(save_file, video_segments, frames, fps=30):
        out = None
        cmap = plt.get_cmap("tab10")
        height, width = frames[0].shape[:2]
        fontsize = max(12, int(min(width, height) / 40))
        font = ImageFont.truetype("default_font_ch_en.ttf", fontsize)
        for i, f in enumerate(frames):
            pil_image = Image.fromarray(f).convert("RGBA")
            if not out:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    save_file, fourcc, fps, (pil_image.size[0], pil_image.size[1])
                )

            for obj_id in video_segments[i]:
                color = list(map(lambda x: int(255 * x), cmap(obj_id)[:3]))
                np_mask = np.zeros((pil_image.size[1], pil_image.size[0], 4))
                mask = video_segments[i][obj_id].mask[0, :, :]
                np_mask[mask > 0, :] = color + [int(255 * 0.6)]
                mask_image = Image.fromarray(np_mask.astype(np.uint8))
                pil_image = Image.alpha_composite(pil_image, mask_image)

                label = video_segments[i][obj_id].label
                label = label + f": {obj_id}"
                draw = ImageDraw.Draw(pil_image)
                text_box = draw.textbbox((0, 0), text=label, font=font)
                x, y = get_text_coords_from_mask(
                    mask,
                    v_gap=(text_box[3] - text_box[1]) + 10,
                    h_gap=((text_box[2] - text_box[0]) // 2),
                )
                if x != 0 and y != 0:
                    text_box = draw.textbbox((x, y), text=label, font=font)
                    draw.rectangle((x, y, text_box[2], text_box[3]), fill=tuple(color))
                    draw.text((x, y), label, fill="black", font=font)

            out.write(
                cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_BGR2RGB)
            )
        if out is not None:
            out.release()

    frames = read_frames(video_path)
    masks_to_video("save.mp4", output, frames)
