import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from pydantic import BaseModel, Field, validate_call
from vision_agent_tools.shared_types import BaseMLModel, Device, VideoNumpy
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Config(BaseModel):
    model_name: str = Field(
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
    def __init__(self, model_config: Optional[SAM2Config] = None):
        self.model_config = model_config or SAM2Config()
        self.image_model = SAM2ImagePredictor.from_pretrained(
            self.model_config.model_name
        )

    @torch.inference_mode()
    def predict_image(
        self,
        image: Image.Image,
        input_box: Optional[np.ndarray] = None,
        input_points: Optional[np.ndarray] = None,
        input_label: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the input image with the SAM2 image predictor using the given prompts.

        Args:
            image (Image.Image): Input image to be processed.
            input_box (Optional[np.ndarray]): Coordinates for boxes.
            input_points (Optional[np.ndarray]): Coordinates for points.
            input_label (Optional[np.ndarray]): Labels for the points.
            multimask_output (bool): Flag whether to output multiple masks.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Detected masks, scores, and logits.
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
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            masks, scores, logits = self.image_model.predict(
                point_coords=input_points,
                point_labels=input_label,
                box=input_box,
                multimask_output=multimask_output,
            )

            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

        return masks, scores, logits

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        input_box: Optional[np.ndarray] = None,
        input_points: Optional[np.ndarray] = None,
        input_label: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ):
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            return self.predict_image(
                image,
                point_coords=input_points,
                point_labels=input_label,
                box=input_box,
                multimask_output=multimask_output,
            )
        elif video is not None:
            raise ValueError("Video not implemented yet.")

    def to(self, device: Device):
        self.image_model.to(device=device.value)
