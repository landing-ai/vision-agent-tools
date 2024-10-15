import os
import wget
import gdown
import os.path as osp
from vision_agent_tools.shared_types import (
    BoundingBox,
    SegmentationBitMask,
    BboxLabel,
    FlorenceV2ODRes,
)
import numpy as np


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
CHECKPOINT_DIR = osp.join(CURRENT_DIR, "checkpoints")


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        if url.startswith("https://drive.google.com"):
            gdown.download(url, path, quiet=False, fuzzy=True)
        else:
            wget.download(url, out=path)
    return path


def calculate_mask_iou(mask1: SegmentationBitMask, mask2: SegmentationBitMask) -> float:
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
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    rows, cols = np.where(mask)
    if len(rows) > 0 and len(cols) > 0:
        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)
        return [x_min, y_min, x_max, y_max]


def convert_florence_bboxes_to_bbox_labels(
    predictions: FlorenceV2ODRes,
) -> list[BboxLabel]:
    """
    Converts the output of the Florecev2 <OD> an
    <CAPTION_TO_PHRASE_GROUNDING> tasks
    to a much simpler list of BboxLabel labels
    """
    od_response = [
        BboxLabel(
            bbox=predictions.bboxes[i],
            label=predictions.labels[i],
            score=1.0,  # FlorenceV2 doesn't provide confidence score
        )
        for i in range(len(predictions.labels))
    ]
    return od_response


def convert_bbox_labels_to_florence_bboxes(predictions: list[BboxLabel]) -> dict:
    """
    Converts the simpler list of BboxLabel labels  format to the format
    of Florecev2 OD and CAPTION_TO_PHRASE_GROUNDING task output.
    """
    preds = {
        "bboxes": [predictions[i].bbox for i in range(len(predictions))],
        "labels": [predictions[i].label for i in range(len(predictions))],
    }
    return preds
