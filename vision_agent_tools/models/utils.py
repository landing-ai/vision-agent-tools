import os
import os.path as osp

import wget
import gdown
import torch
import numpy as np

from vision_agent_tools.shared_types import (
    BoundingBox,
    SegmentationBitMask,
    Device
)


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
CHECKPOINT_DIR = osp.join(CURRENT_DIR, "checkpoints")


def get_device() -> Device:
    return (
        Device.GPU
        if torch.cuda.is_available()
        else Device.MPS if torch.backends.mps.is_available() else Device.CPU
    )


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        if url.startswith("https://drive.google.com"):
            gdown.download(url, path, quiet=False, fuzzy=True)
        else:
            wget.download(url, out=path)
    return path


def calculate_mask_iou(mask1: SegmentationBitMask, mask2: SegmentationBitMask) -> float:
    """Calculate the Intersection over Union (IoU) between two masks.

    Parameters:
        mask1:
            First mask.
        mask2:
            Second mask.

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


def calculate_bbox_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        bbox1:
            First bounding box [x_min, y_min, x_max, y_max].
        bbox2:
            Second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        float: IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    x_max_inter = min(bbox1[2], bbox2[2])
    y_max_inter = min(bbox1[3], bbox2[3])

    # Calculate the area of the intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate the IoU
    return inter_area / union_area if union_area != 0 else 0
