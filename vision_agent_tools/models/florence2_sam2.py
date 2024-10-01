import torch
import numpy as np
from PIL import Image
from pydantic import BaseModel, validate_call, Field
from typing import Annotated

from vision_agent_tools.shared_types import (
    BaseMLModel,
    VideoNumpy,
    Device,
    BboxAndMaskLabel,
    FlorenceV2ODRes,
)
from vision_agent_tools.models.florencev2 import Florencev2, PromptTask

from vision_agent_tools.models.utils import (
    calculate_mask_iou,
    mask_to_bbox,
    convert_florence_bboxes_to_bbox_labels,
)

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Florence2SAM2Config(BaseModel):
    hf_model: str = Field(
        default="facebook/sam2-hiera-large",
        description="Name of the model",
    )
    device: Device = Field(
        default=Device.GPU
        if torch.cuda.is_available()
        else Device.MPS
        if torch.backends.mps.is_available()
        else Device.CPU,
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. Default is the first available GPU.",
    )


class Florence2SAM2(BaseMLModel):
    """
    A class that receives a video or an image plus a list of text prompts and
    returns the instance segmentation for the text prompts in each frame.
    """

    def __init__(self, model_config: Florence2SAM2Config | None = None):
        """
        Initializes the Florence2SAM2 object with a pre-trained Florencev2 model
        and a SAM2 model.
        """
        self._model_config = model_config or Florence2SAM2Config()
        self.florence2 = Florencev2()
        self.video_predictor = SAM2VideoPredictor.from_pretrained(
            self._model_config.hf_model
        )
        self.image_predictor = SAM2ImagePredictor(self.video_predictor)

    def _update_reference_predictions(
        self,
        last_predictions: list[BboxAndMaskLabel],
        new_predictions: list[BboxAndMaskLabel],
        objects_count: int,
        iou_threshold: float = 0.6,
    ) -> tuple[list[BboxAndMaskLabel], int]:
        """
        Updates the object prediction ids of the 'new_predictions' input to match
        the ids coming from the 'last_predictions' input, by comparing the IoU between
        the annotation of the two lists.

        Parameters:
        last_predictions (list[BboxAndMaskLabel]): List containing the
            annotation predictions of the last frame's prediction
            of the video propagation.
        new_predictions (list[BboxAndMaskLabel]): List containing the
            annotation predictions of the FlorenceV2 model prediction.
        iou_threshold (float): The IoU threshold value used to compare last_predictions
            and new_predictions annotation lists.

        Returns:
        tuple[list[BboxAndMaskLabel], int]: The first element of the tuple is
            the updated list of annotations, the second element is the updated object count.
        """
        new_annotation_predictions: list[BboxAndMaskLabel] = []
        for i, new_prediction in enumerate(new_predictions):
            new_ann_id: int = 0
            for last_prediction in last_predictions:
                iou = calculate_mask_iou(
                    new_prediction.mask,
                    last_prediction.mask,
                )
                if iou > iou_threshold:
                    new_ann_id = last_prediction.id
                    break

            if not new_ann_id:
                objects_count += 1
                new_ann_id = objects_count
                new_predictions[i].id = new_ann_id
                new_annotation_predictions.append(new_predictions[i])

        updated_predictions = last_predictions + new_annotation_predictions
        return (updated_predictions, objects_count)

    @torch.inference_mode()
    def _get_bbox_and_mask(
        self,
        prompt: str,
        image: Image.Image,
        return_mask: bool = True,
        nms_threshold: float = 1.0,
    ) -> list[BboxAndMaskLabel]:
        annotations: list[BboxAndMaskLabel] = []
        self.image_predictor.set_image(np.array(image, dtype=np.uint8))
        with torch.autocast(device_type=self._model_config.device, dtype=torch.float16):
            preds = self.florence2(
                image=image,
                task=PromptTask.CAPTION_TO_PHRASE_GROUNDING,
                prompt=prompt,
                nms_threshold=nms_threshold,
            )[PromptTask.CAPTION_TO_PHRASE_GROUNDING]
        preds = convert_florence_bboxes_to_bbox_labels(FlorenceV2ODRes(**preds))
        if return_mask:
            with torch.autocast(
                device_type=self._model_config.device, dtype=torch.bfloat16
            ):
                masks, _, _ = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=[elt.bbox for elt in preds],
                    multimask_output=False,
                )
        for i, pred in enumerate(preds):
            annotations.append(
                BboxAndMaskLabel(
                    bbox=pred.bbox,
                    mask=(
                        masks[i, 0, :, :] if len(masks.shape) == 4 else masks[i, :, :]
                    )
                    if return_mask
                    else None,
                    label=pred.label,
                    score=1.0,
                    id=i,
                )
            )
        return annotations

    @torch.inference_mode()
    def handle_image(
        self,
        prompt: str,
        image: Image.Image,
        nms_threshold: float = 1.0,
    ) -> list[list[BboxAndMaskLabel]]:
        self.image_predictor.reset_predictor()
        objs = self._get_bbox_and_mask(
            prompt, image.convert("RGB"), nms_threshold=nms_threshold
        )
        return [objs]

    @torch.inference_mode()
    def handle_video(
        self,
        prompt: str,
        video: VideoNumpy,
        chunk_length: int | None = 20,
        iou_threshold: float = 0.6,
        nms_threshold: float = 1.0,
    ) -> list[list[BboxAndMaskLabel]]:
        video_shape = video.shape
        num_frames = video_shape[0]
        video_segments: list[list[BboxAndMaskLabel]] = []
        objects_count = 0
        last_chunk_frame_pred: list[BboxAndMaskLabel] = []

        if chunk_length is None:
            chunk_length = num_frames
        with torch.autocast(
            device_type=self._model_config.device, dtype=torch.bfloat16
        ):
            inference_state = self.video_predictor.init_state(video=video)

            for start_frame_idx in range(0, num_frames, chunk_length):
                self.image_predictor.reset_predictor()
                new_frame_preds = self._get_bbox_and_mask(
                    prompt,
                    Image.fromarray(video[start_frame_idx]).convert("RGB"),
                    nms_threshold=nms_threshold,
                )
                # Compare the IOU between the predicted label 'new_frame_preds' and the 'last_chunk_frame_pred'
                # and update the object prediction id, to match the previous id.
                # Also add the new objects in case they didn't exist before.
                updated_objs, objects_count = self._update_reference_predictions(
                    last_chunk_frame_pred,
                    new_frame_preds,
                    objects_count,
                    iou_threshold,
                )
                self.video_predictor.reset_state(inference_state)

                # Add new label points to the video predictor coming from the FlorenceV2 object predictions
                annotation_id_to_label = {}
                for updated_obj in updated_objs:
                    annotation_id = updated_obj.id
                    annotation_id_to_label[annotation_id] = updated_obj.label
                    _, _, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=start_frame_idx,
                        obj_id=annotation_id,
                        box=updated_obj.bbox,
                    )
                # Propagate the predictions on the given video segment (chunk)
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.video_predictor.propagate_in_video(
                    inference_state, start_frame_idx, chunk_length
                ):
                    if out_frame_idx not in range(len(video_segments)):
                        video_segments.append([])

                    for i, out_obj_id in enumerate(out_obj_ids):
                        pred_mask = (out_mask_logits[i][0] > 0.0).cpu().numpy()
                        if np.max(pred_mask) == 0:
                            continue
                        video_segments[out_frame_idx].append(
                            BboxAndMaskLabel(
                                label=annotation_id_to_label[out_obj_id],
                                bbox=mask_to_bbox(pred_mask),
                                mask=pred_mask,
                                score=1.0,
                                id=out_obj_id,
                            )
                        )
                index = (
                    start_frame_idx + chunk_length
                    if (start_frame_idx + chunk_length) < num_frames
                    else num_frames - 1
                )
                # Save the last frame predictions to later update the newly found FlorenceV2 object ids
                last_chunk_frame_pred = video_segments[index]
                self.video_predictor.reset_state(inference_state)

        return video_segments

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        chunk_length: int | None = 20,
        iou_threshold: Annotated[float, Field(ge=0.1, le=1.0)] = 0.6,
        nms_threshold: Annotated[float, Field(ge=0.1, le=1.0)] = 1.0,
    ) -> list[list[BboxAndMaskLabel]]:
        """
        Florence2Sam2 model find objects in an image and track objects in a video.

        Args:
            prompt (list[str]): The list of objects to be found.
            image (Image.Image | None): The image to be analyzed.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            chunk_length (int): The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.
            iou_threshold (float): The IoU threshold value used to compare last_predictions and new_predictions objects.
            nms_threshold (float): The non-maximum suppression threshold value used to filter the Florencev2 predictions.

        Returns:
            list[list[ImageBboxMaskLabel]]: a list where the first list contains the frames predictions,
            then the second list contains the annotation, where the annotations are objects with the mask,
            label and bbox (for images) for each annotation. For example:
                [
                    [
                        BboxAndMaskLabel({"mask": np.ndarray, "label": "car", score: 0.9}),
                        BboxAndMaskLabel({"mask", np.ndarray, "label": "person", score: 0.8}),
                    ],
                    ...
                ]
        """

        prompt = ", ".join(prompts)
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            return self.handle_image(prompt, image, nms_threshold)
        elif video is not None:
            assert video.ndim == 4, "Video should have 4 dimensions"
            return self.handle_video(
                prompt, video, chunk_length, iou_threshold, nms_threshold
            )
        # No need to raise an error here, the validatie_call decorator will take care of it
