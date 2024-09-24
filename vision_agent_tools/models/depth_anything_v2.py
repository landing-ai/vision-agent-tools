import os

# Run this line before loading torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os.path as osp
from typing import Any, Union

import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Model
from PIL import Image
from pydantic import BaseModel

from vision_agent_tools.shared_types import BaseMLModel, Device

from .utils import CHECKPOINT_DIR, download


class DepthMap(BaseModel):
    """Represents the depth map of an image.

    Attributes:
        map (Any): HxW raw depth map of the image.
    """

    map: Any


class DepthAnythingV2(BaseMLModel):
    """
    Model for depth estimation using the Depth-Anything-V2 model from the paper
    [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

    """

    _CHECKPOINT_DIR = CHECKPOINT_DIR

    def __init__(self) -> None:
        """
        Initializes the Depth-Anything-V2 model.
        """
        if not osp.exists(self._CHECKPOINT_DIR):
            os.makedirs(self._CHECKPOINT_DIR)

        DEPTH_ANYTHING_V2_CHECKPOINT = (
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
            "depth_anything_v2_vits.pth",
        )
        # init model
        self._model = DepthAnythingV2Model(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model_checkpoint_path = download(
            url=DEPTH_ANYTHING_V2_CHECKPOINT[0],
            path=os.path.join(self._CHECKPOINT_DIR, DEPTH_ANYTHING_V2_CHECKPOINT[1]),
        )

        state_dict = torch.load(self.model_checkpoint_path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def __call__(
        self, image: Union[str, Image.Image], grayscale: bool | None = False
    ) -> DepthMap:
        """Depth-Anything-V2 is a highly practical solution for robust monocular depth estimation.

        Args:
            image (Union[str, Image.Image, np.ndarray]): The input image for depth estimation.
                Can be a file path, a PIL Image, or a NumPy array.
            grayscale (bool, optional): Whether to return the depth map as a grayscale image.
                If True, the depth map will be normalized to the range [0, 255] and converted
                to uint8. Defaults to False.

        Returns:
            DepthMap: An object type containing a numpy array with the HxW depth map of the image.
        """
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        depth_map = self._model.infer_image(image)  # HxW raw depth map

        if grayscale:
            # Normalize depth map to [0, 255] and convert to uint8
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map_normalized = (depth_map - depth_min) / (
                depth_max - depth_min + 1e-8
            )
            depth_map = (255 * depth_map_normalized).astype(np.uint8)

        return DepthMap(map=depth_map)

    def to(self, device: Device):
        self._model.to(device=device.value)
