import os.path as osp
import os
import torch

from PIL import Image
from loca.loca import LOCA
from .utils import download
from typing import Union, List, Optional, Any
from torch import nn
from torchvision import transforms as T
from pydantic import BaseModel
from vision_agent_tools.tools.shared_types import BaseTool


class CountingDetection(BaseModel):
    count: int
    masks: List[Any]


class ZeroShotCounting(BaseTool):
    _CURRENT_DIR = osp.dirname(osp.abspath(__file__))
    _TARGET_DIR = f"{_CURRENT_DIR}/counting_model"
    _CHECKPOINT_DIR = osp.join(_TARGET_DIR, "checkpoint")

    def __init__(self, img_size=512) -> None:
        if not osp.exists(self._CHECKPOINT_DIR):
            os.makedirs(self._CHECKPOINT_DIR)

        ZSHOT_CHECKPOINT = (
            "https://drive.google.com/file/d/11-gkybBmBhQF2KZyo-c2-4IGUmor_JMu/view?usp=sharing",
            "count_zero_shot.pt",
        )

        # init model
        self._model = LOCA(
            image_size=img_size,
            num_encoder_layers=3,
            num_ope_iterative_steps=3,
            num_objects=3,
            zero_shot=True,
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

        # Check if CUDA (GPU support) is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(
                "Using GPU:", torch.cuda.get_device_name(0)
            )  # Prints the GPU device name
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model_checkpoint_path = download(
            url=ZSHOT_CHECKPOINT[0],
            path=os.path.join(self._CHECKPOINT_DIR, ZSHOT_CHECKPOINT[1]),
        )

        state_dict = torch.load(
            self.model_checkpoint_path, map_location=torch.device(self.device)
        )["model"]
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._model.to(self.device)
        self.img_size = img_size

    @torch.no_grad()
    def __call__(
        self, image: Union[str, Image.Image], bbox: Optional[List[float]] = None
    ) -> CountingDetection:
        if bbox:
            assert len(bbox) == 4, "Bounding box should be in format [x1, y1, x2, y2]"
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
        dmap = dmap.squeeze().cpu().numpy().astype("uint8")
        density_map = Image.fromarray(dmap, mode="L")
        return CountingDetection(count=round(n_objects), masks=[density_map])

    def to(self, device):
        self._model.to(device)
