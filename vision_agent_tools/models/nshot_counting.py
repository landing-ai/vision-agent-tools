import os

# Run this line before loading torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os.path as osp
from typing import Any, Optional

import torch
from loca.loca import LOCA
from PIL import Image
from pydantic import BaseModel
from torch import nn
from torchvision import transforms as T

from vision_agent_tools.shared_types import BaseMLModel, Device, BoundingBox

from .utils import CHECKPOINT_DIR, download


class CountingDetection(BaseModel):
    """
    Represents an inference result from the LOCA model.

    Attributes:
        count (int): The predicted number of detected objects.
        masks (list[Any]): A list of numpy arrays representing the masks
                        of the detected objects in the image.
    """

    count: int
    heat_map: Any


class NShotCounting(BaseMLModel):
    """
    Model for object counting using the zeroshot and n-shot versions of the LOCA model from the paper
    [A Low-Shot Object Counting Network With Iterative Prototype Adaptation ](https://github.com/djukicn/loca).

    """

    _CHECKPOINT_DIR = CHECKPOINT_DIR

    def __init__(self, zero_shot=True, img_size=512) -> None:
        """
        Initializes the LOCA model.

        Args:
            img_size (int): Size of the input image.

        """
        if not osp.exists(self._CHECKPOINT_DIR):
            os.makedirs(self._CHECKPOINT_DIR)

        ZSHOT_CHECKPOINT = (
            "https://drive.google.com/file/d/11-gkybBmBhQF2KZyo-c2-4IGUmor_JMu/view?usp=sharing",
            "count_zero_shot.pt",
        )
        FSHOT_CHECKPOINT = (
            "https://drive.google.com/file/d/1rTG7AjGmasfOYFm-ZzSbVQH9daYgOoIS/view?usp=sharing",
            "count_few_shot.pt",
        )

        # init model
        self._model = LOCA(
            image_size=img_size,
            num_encoder_layers=3,
            num_ope_iterative_steps=3,
            num_objects=3 if zero_shot else 1,
            zero_shot=zero_shot,
            emb_dim=256,
            num_heads=8,
            kernel_dim=3,
            backbone_name="resnet50",
            swav_backbone=True,
            train_backbone=False,
            reduction=8,
            dropout=0.1,
            layer_norm_eps=1e-5,
            mlp_factor=8,
            norm_first=True,
            activation=nn.GELU,
            norm=True,
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        if zero_shot:
            self.model_checkpoint_path = download(
                url=ZSHOT_CHECKPOINT[0],
                path=os.path.join(self._CHECKPOINT_DIR, ZSHOT_CHECKPOINT[1]),
            )
        else:
            self.model_checkpoint_path = download(
                url=FSHOT_CHECKPOINT[0],
                path=os.path.join(self._CHECKPOINT_DIR, FSHOT_CHECKPOINT[1]),
            )

        state_dict = torch.load(
            self.model_checkpoint_path, map_location=torch.device(self.device)
        )["model"]
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()
        self.img_size = img_size

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        bbox: Optional[BoundingBox] = None,
    ) -> CountingDetection:
        """
        LOCA injects shape and appearance information into object queries
        to precisely count objects of various sizes in densely and sparsely populated scenarios.
        It also extends to a zeroshot scenario and achieves excellent localization and count errors
        across the entire low-shot spectrum.

        Args:
            image (Image.Image): The input image for object detection.
            bbox (BoundingBox): A list of four ints representing the bounding box coordinates (xmin, ymin, xmax, ymax)
                        of the detected query in the image.

        Returns:
            CountingDetection: An object type containing:
                - The count of the objects found similar to the bbox query.
                - A list of numpy arrays representing the masks of the objects found.
        """
        if bbox:
            assert len(bbox) == 4, "Bounding box should be in format [x1, y1, x2, y2]"
        image = image.convert("RGB")
        w, h = image.size
        img_t = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.img_size, self.img_size)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(image).to(self.device)
        if bbox:
            bbox = (torch.tensor(bbox) / torch.tensor([w, h, w, h]) * self.img_size).to(
                self.device
            )
        else:
            bbox = torch.ones(2, device=self.device)

        out, _ = self._model(img_t[None], bbox[None].unsqueeze(0))

        n_objects = out.flatten(1).sum(dim=1).cpu().numpy().item()

        dmap = (out - torch.min(out)) / (torch.max(out) - torch.min(out)) * 255
        density_map = dmap.squeeze().cpu().numpy().astype("uint8")
        return CountingDetection(count=round(n_objects), heat_map=[density_map])

    def to(self, device: Device):
        self._model.to(device=device.value)
