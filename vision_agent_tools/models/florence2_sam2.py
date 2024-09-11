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
    bounding_box: (
        list[
            Annotated[float, "x_min"],
            Annotated[float, "y_min"],
            Annotated[float, "x_max"],
            Annotated[float, "y_max"],
        ]
        | None
    )
    mask: SegmentationBitMask | None


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

    def _calculate_iou(
        self, mask1: SegmentationBitMask, mask2: SegmentationBitMask
    ) -> float:
        """
        Calculate the Intersection over Union (IoU) between two masks.

        Parameters:
        mask1 (numpy.ndarray): First mask.
        mask2 (numpy.ndarray): Second mask.

        Returns:
        float: IoU value.
        """
        # Ensure the masks are binary
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)

        # Calculate the intersection and union
        intersection = np.sum(np.logical_and(mask1, mask2))
        union = np.sum(np.logical_or(mask1, mask2))

        # Calculate the IoU
        iou = intersection / union if union != 0 else 0

        return iou

    def _mask_to_bbox(self, mask: np.ndarray):
        rows, cols = np.where(mask)
        if len(rows) > 0 and len(cols) > 0:
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
            return [x_min, y_min, x_max, y_max]

    def _update_reference_predictions(
        self,
        last_predictions: dict[int, ImageBboxAndMaskLabel],
        new_predictions: dict[int, ImageBboxAndMaskLabel],
        objects_count: int,
        iou_threshold: float = 0.8,
    ) -> tuple[dict[int, ImageBboxAndMaskLabel], int]:
        """
        Updates the object prediction ids of the 'new_predictions' input to match
        the ids coming from the 'last_predictions' input, by comparing the IoU between
        the two elements.

        Parameters:
        last_predictions (dict[int, ImageBboxAndMaskLabel]): Dictionary containing the
            id of the object as the key and the prediction as the value of the last frame's prediction
            of the video propagation.
        new_predictions (dict[int, ImageBboxAndMaskLabel]): Dictionary containing the
            id of the object as the key and the prediction as the value of the FlorenceV2 model prediction.
        iou_threshold (float): The IoU threshold value used to compare last_predictions and new_predictions objects.

        Returns:
        float: IoU value.
        """
        updated_predictions: dict[int, ImageBboxAndMaskLabel] = {}
        for new_annotation_id in new_predictions:
            new_obj_id: int = 0
            for old_annotation_id in last_predictions:
                iou = self._calculate_iou(
                    new_predictions[new_annotation_id].mask,
                    last_predictions[old_annotation_id].mask,
                )
                if iou > iou_threshold:
                    new_obj_id = old_annotation_id
                    updated_predictions[new_obj_id] = ImageBboxAndMaskLabel(
                        bounding_box=last_predictions[new_obj_id].bounding_box,
                        mask=last_predictions[new_obj_id].mask,
                        label=last_predictions[new_obj_id].label,
                    )
                    break

            if not new_obj_id:
                objects_count += 1
                new_obj_id = objects_count
                updated_predictions[new_obj_id] = new_predictions[new_annotation_id]

        return (updated_predictions, objects_count)

    @torch.inference_mode()
    def _get_bbox_and_mask(
        self, prompt: str, image: Image.Image, return_mask: bool = True
    ) -> dict[int, ImageBboxAndMaskLabel]:
        objs = {}
        self.image_predictor.set_image(np.array(image, dtype=np.uint8))
        annotation_id = 0
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            preds = self.florence2(
                image=image,
                task=PromptTask.CAPTION_TO_PHRASE_GROUNDING,
                prompt=prompt,
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
        objs = self._get_bbox_and_mask(prompt, image.convert("RGB"))
        return {0: objs}

    @torch.inference_mode()
    def handle_video(
        self,
        prompt: str,
        video: VideoNumpy,
        chunk_length: int | None = 20,
        iou_threshold: float = 0.8,
    ) -> tuple[dict[int, dict[int, ImageBboxAndMaskLabel]], dict[int, dict[int, ImageBboxAndMaskLabel]], dict[int, dict[int, ImageBboxAndMaskLabel]]]:
        video_shape = video.shape
        num_frames = video_shape[0]
        video_segments = {}
        objects_count = 0
        sam2_preds: dict[int, dict[int, ImageBboxAndMaskLabel]] = {}
        florence2_preds: dict[int, dict[int, ImageBboxAndMaskLabel]] = {}
        last_chunk_frame_pred: dict[int, ImageBboxAndMaskLabel] = {}

        if chunk_length is None:
            chunk_length = num_frames
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            inference_state = self.video_predictor.init_state(video=video)

            for start_frame_idx in range(0, num_frames, chunk_length):
                self.image_predictor.reset_predictor()
                # fl_frame_idx = 0 if len(last_chunk_frame_pred.keys()) == 0 else start_frame_idx - 1
                print("start_frame_idx: ", start_frame_idx)
                objs = self._get_bbox_and_mask(
                    prompt,
                    Image.fromarray(video[start_frame_idx]).convert("RGB"),
                )
                florence2_preds[start_frame_idx] = objs
                # Compare the IOU between the predicted label 'objs' and the 'last_chunk_frame_pred'
                # and update the object prediction id, to match the previous id.
                # Also add the new objects in case they didn't exist before.
                updated_objs, objects_count = self._update_reference_predictions(
                    last_chunk_frame_pred, objs, objects_count, iou_threshold
                )
                self.video_predictor.reset_state(inference_state)

                # Add new label points to the video predictor coming from the FlorenceV2 object predictions
                annotation_id_to_label = {}
                for annotation_id in updated_objs:
                    annotation_id_to_label[annotation_id] = updated_objs[
                        annotation_id
                    ].label
                    _, _, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=start_frame_idx,
                        obj_id=annotation_id,
                        box=updated_objs[annotation_id].bounding_box,
                    )
                # Propagate the predictions on the given video segment (chunk)
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.video_predictor.propagate_in_video(
                    inference_state, start_frame_idx, chunk_length
                ):
                    if out_frame_idx not in video_segments.keys():
                        video_segments[out_frame_idx] = {}

                    for i, out_obj_id in enumerate(out_obj_ids):
                        pred_mask = (out_mask_logits[i][0] > 0.0).cpu().numpy()
                        video_segments[out_frame_idx][out_obj_id] = (
                            ImageBboxAndMaskLabel(
                                label=annotation_id_to_label[out_obj_id],
                                bounding_box=self._mask_to_bbox(pred_mask),
                                mask=pred_mask,
                            )
                        )
                index = (
                    start_frame_idx + chunk_length
                    if (start_frame_idx + chunk_length) < num_frames
                    else num_frames - 1
                )
                # Save the last frame predictions to later update the newly found FlorenceV2 object ids
                last_chunk_frame_pred = video_segments[index]
                sam2_preds[index] = video_segments[index]
                self.video_predictor.reset_state(inference_state)

        return (video_segments, sam2_preds, florence2_preds)

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        chunk_length: int | None = 20,
        iou_threshold: float = 0.8,
    ) -> dict[int, dict[int, ImageBboxAndMaskLabel]]:
        """
        Florence2Sam2 model find objects in an image and track objects in a video.

        Args:
            prompt (list[str]): The list of objects to be found.
            image (Image.Image | None): The image to be analyzed.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            chunk_length (int): The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.

        Returns:
            dict[int, ImageBboxMaskLabel]: a dictionary where the first key is the frame index
            then an annotation ID, then an object with the mask, label and possibly bbox (for images)
            for each annotation ID. For example:
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
            return self.handle_video(prompt, video, chunk_length, iou_threshold)
        # No need to raise an error here, the validatie_call decorator will take care of it
