from dataclasses import dataclass
from typing_extensions import Annotated

import torch
import numpy as np
from PIL import Image
from pydantic import validate_call

from vision_agent_tools.shared_types import BaseMLModel, VideoNumpy, SegmentationBitMask
from vision_agent_tools.models.florencev2 import Florencev2, PromptTask

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


_HF_MODEL = "facebook/sam2-hiera-large"


@dataclass
class ImageBboxAndMaskLabel:
    label: str
    bounding_box: list[
        Annotated[float, "x_min"],
        Annotated[float, "y_min"],
        Annotated[float, "x_max"],
        Annotated[float, "y_max"],
    ]
    mask: SegmentationBitMask | None


@dataclass
class MaskLabel:
    label: str
    mask: SegmentationBitMask


class Florence2SAM2(BaseMLModel):
    """
    A class that receives a video or an image plus a list of text prompts and
    returns the instance segmentation for the text prompts in each frame.
    """

    def __init__(self, device: str | None = None):
        """
        Initializes the Florence2SAM2 object with a pre-trained Florencev2 model
        and a SAM2 model.
        """
        self.device = (
            device
            if device in ["cuda", "mps", "cpu"]
            else "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.florence2 = Florencev2()
        self.video_predictor = SAM2VideoPredictor.from_pretrained(_HF_MODEL)
        self.image_predictor = SAM2ImagePredictor(self.video_predictor)

    @torch.inference_mode()
    def get_bbox_and_mask(
        self, prompt: str, image: Image.Image, return_mask: bool = True
    ) -> dict[int, ImageBboxAndMaskLabel]:
        objs = {}
        self.image_predictor.set_image(np.array(image, dtype=np.uint8))
        annotation_id = 0
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            preds = self.florence2(
                image=image, task=PromptTask.CAPTION_TO_PHRASE_GROUNDING, prompt=prompt,
            )[PromptTask.CAPTION_TO_PHRASE_GROUNDING]
        preds = [
            {"bbox": preds["bboxes"][i], "label": preds["labels"][i]}
            for i in range(len(preds["labels"]))
        ]
        if return_mask:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                masks, _, _ = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=[elt["bbox"] for elt in preds],
                    multimask_output=False,
                )
        for i in range(len(preds)):
            objs[annotation_id] = ImageBboxAndMaskLabel(
                bounding_box=preds[i]["bbox"],
                mask=(masks[i, 0, :, :] if len(masks.shape) == 4 else masks[i, :, :])
                if return_mask
                else None,
                label=preds[i]["label"],
            )
            annotation_id += 1
        return objs

    @torch.inference_mode()
    def handle_image(
        self,
        prompt: str,
        image: Image.Image,
    ) -> dict[int, dict[int, ImageBboxAndMaskLabel]]:
        self.image_predictor.reset_predictor()
        objs = self.get_bbox_and_mask(prompt, image.convert("RGB"))
        return {0: objs}

    @torch.inference_mode()
    def handle_video(
        self,
        prompt: str,
        video: VideoNumpy,
        step: int = 20,
    ) -> tuple[
        dict[int, dict[int, MaskLabel]], dict[int, dict[int, ImageBboxAndMaskLabel]]
    ]:
        video_shape = video.shape
        video_segments = {}
        image_predictions = {}

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            inference_state = self.video_predictor.init_state(video=video)

            for start_frame_idx in range(0, video_shape[0], step):
                self.image_predictor.reset_predictor()
                objs = self.get_bbox_and_mask(
                    prompt,
                    Image.fromarray(video[start_frame_idx]).convert("RGB"),
                    return_mask=False,
                )
                image_predictions[start_frame_idx] = objs
                # prompt grounding dino to get the box coordinates on specific frame
                # print("start_frame_idx", start_frame_idx)
                self.video_predictor.reset_state(inference_state)
                annotation_id_to_label = {}
                for annotation_id in objs:
                    annotation_id_to_label[annotation_id] = objs[annotation_id].label
                    _, _, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=start_frame_idx,
                        obj_id=annotation_id,
                        box=objs[annotation_id].bounding_box,
                    )

                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.video_predictor.propagate_in_video(
                    inference_state, start_frame_idx, step
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: MaskLabel(
                            mask=(out_mask_logits[i][0] > 0.0).cpu().numpy(),
                            label=annotation_id_to_label[out_obj_id],
                        )
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                self.video_predictor.reset_state(inference_state)
        return (video_segments, image_predictions)

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        step: int | None = 20,
    ) -> dict[int, dict[int, ImageBboxAndMaskLabel | MaskLabel]]:
        """Returns a dictionary where the first key is the frame index then an annotation
        ID, then an object with the mask, label and possibly bbox (for images) for each
        annotation ID. For example:
        {
            0:
                {
                    0: ImageBboxMaskLabel({"mask": np.ndarray, "label": "car"}),
                    1: ImageBboxMaskLabel({"mask", np.ndarray, "label": "person"})
                },
            1: ...
        }
        """

        prompt = ", ".join(prompts)
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            return self.handle_image(prompt, image)
        elif video is not None:
            assert video.ndim == 4, "Video should have 4 dimensions"
            return self.handle_video(prompt, video, step)
        # No need to raise an error here, the validatie_call decorator will take care of it
