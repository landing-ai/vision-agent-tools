import torch
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, validate_call
from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


class SAM2Config(BaseModel):
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


class SAM2Model(BaseMLModel):
    def __init__(self, model_config: SAM2Config | None = None):
        self.model_config = model_config or SAM2Config()
        self.image_model = SAM2ImagePredictor.from_pretrained(
            self.model_config.hf_model
        )
        self.video_model = SAM2VideoPredictor.from_pretrained(
            self.model_config.hf_model
        )

    @torch.inference_mode()
    def predict_image(
        self,
        image: Image.Image,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
        multimask_output: bool = False,
    ):
        """
        Process the input image with the SAM2 image predictor using the given prompts.

        Args:
            image (Image.Image): Input image to be processed.
            input_box (np.ndarray): Coordinates for boxes.
            input_points (np.ndarray): Coordinates for points.
            input_label (np.ndarray): Labels for the points.
            multimask_output (bool): Flag whether to output multiple masks.

        """
        if input_box is None and input_points is None and input_label is None:
            raise ValueError(
                "Either 'input_box' or 'input_points' and 'input_label' must be provided."
            )

        self.image_model.reset_predictor()
        np_image = np.array(image.convert("RGB"), dtype=np.uint8)
        self.image_model.set_image(np_image)

        torch_dtype = (
            torch.bfloat16 if self.model_config.device == Device.GPU else torch.float16
        )

        masks, scores, logits = None, None, None

        with torch.autocast(device_type=self.model_config.device, dtype=torch_dtype):
            if (
                self.model_config.device == Device.GPU
                and torch.cuda.get_device_properties(0).major >= 8
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            masks, scores, logits = self.image_model.predict(
                point_coords=input_points,
                point_labels=input_label,
                box=input_box,
                multimask_output=multimask_output,
            )

        return masks, scores, logits

    @torch.inference_mode()
    def predict_video(
        self,
        video: VideoNumpy,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
    ):
        """
        Process the input video with the SAM2 video predictor using the given prompts.

        Args:
            video (VideoNumpy): Input video to be processed.
            input_box (np.ndarray): Coordinates for boxes.
            input_points (np.ndarray): Coordinates for points.
            input_label (np.ndarray): Labels for the points.

        Returns:
            dict: A dictionary mapping frame indices to a list of masks for each object detected in that frame.
        """
        if input_box is None and input_points is None and input_label is None:
            raise ValueError(
                "Either 'input_box' or 'input_points' and 'input_label' must be provided."
            )

        torch_dtype = (
            torch.bfloat16 if self.model_config.device == Device.GPU else torch.float16
        )

        inference_state = self.video_model.init_state(video)

        video_segments = {}

        with torch.autocast(device_type=self.model_config.device, dtype=torch_dtype):
            if (
                self.model_config.device == Device.GPU
                and torch.cuda.get_device_properties(0).major >= 8
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Process each frame in the video
            for frame_idx in range(video.shape[0]):
                np_image = np.array(video[frame_idx], dtype=np.uint8)
                self.image_model.set_image(np_image)
                self.video_model.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                    box=input_box,
                    points=input_points,
                    labels=input_label,
                )

            # Propagate the masklets across the video
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_model.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        self.video_model.reset_state(inference_state)

        return video_segments

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        input_box: np.ndarray | None = None,
        input_points: np.ndarray | None = None,
        input_label: np.ndarray | None = None,
        multimask_output: bool = False,
    ):
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            return self.predict_image(
                image,
                input_box=input_box,
                input_points=input_points,
                input_label=input_label,
                multimask_output=multimask_output,
            )
        elif video is not None:
            assert video.ndim == 4, "Video should have 4 dimensions"
            return self.predict_video(
                video,
                input_box=input_box,
                input_points=input_points,
                input_label=input_label,
            )

    def to(self, device: Device):
        self.model_config.device = device
        self.video_model.to(self.model_config.device)
