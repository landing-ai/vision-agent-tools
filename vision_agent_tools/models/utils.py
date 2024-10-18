import os
import logging
import os.path as osp

import wget
import gdown
import torch
import numpy as np

from vision_agent_tools.shared_types import (
    BboxLabel,
    BoundingBox,
    ODResponse,
    SegmentationBitMask,
    Device,
)

_LOGGER = logging.getLogger(__name__)

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
CHECKPOINT_DIR = osp.join(CURRENT_DIR, "checkpoints")


def get_device() -> Device:
    return (
        Device.GPU
        if torch.cuda.is_available()
        else Device.MPS
        if torch.backends.mps.is_available()
        else Device.CPU
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
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    rows, cols = np.where(mask)
    if len(rows) > 0 and len(cols) > 0:
        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)
        return [x_min, y_min, x_max, y_max]


def convert_florence_bboxes_to_bbox_labels(
    predictions: ODResponse,
) -> list[BboxLabel]:
    """
    Converts the output of the Florence2 <OD> an
    <CAPTION_TO_PHRASE_GROUNDING> tasks
    to a much simpler list of BboxLabel labels
    """
    od_response = [
        BboxLabel(
            bbox=predictions.bboxes[i],
            label=predictions.labels[i],
            score=1.0,  # Florence2 doesn't provide confidence score
        )
        for i in range(len(predictions.labels))
    ]
    return od_response


def _contains(box_a, box_b):
    """
    Checks if box_a fully contains box_b.
    Each box is [x_min, y_min, x_max, y_max].
    """
    x_min_a, y_min_a, x_max_a, y_max_a = box_a
    x_min_b, y_min_b, x_max_b, y_max_b = box_b
    return (
        x_min_a <= x_min_b
        and y_min_a <= y_min_b
        and x_max_a >= x_max_b
        and y_max_a >= y_max_b
    )


def filter_redundant_boxes(response, min_contained=2):
    """
    Filters out redundant bounding boxes that fully contain multiple smaller boxes of the same label.

    Parameters:
        response (dict): Dictionary containing 'bboxes' and 'labels'.
        min_contained (int): Minimum number of contained boxes to consider a box redundant.

    Returns:
        output_data (dict): Dictionary with filtered 'bboxes' and 'labels'.
    """
    bboxes = response["bboxes"]
    labels = response["labels"]

    # Organize boxes by label
    label_to_boxes = {}
    for bbox, label in zip(bboxes, labels):
        label_to_boxes.setdefault(label, []).append(bbox)

    filtered_bboxes = []
    filtered_labels = []

    for label, boxes in label_to_boxes.items():
        n = len(boxes)
        if n < min_contained + 1:
            # Not enough boxes to have redundancies
            filtered_bboxes.extend(boxes)
            filtered_labels.extend([label] * n)
            continue

        # Sort boxes by area descending
        boxes_sorted = sorted(
            boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True
        )
        to_remove = set()

        for i in range(n):
            if i in to_remove:
                continue
            box_a = boxes_sorted[i]
            contained = 0
            for j in range(n):
                if i == j or j in to_remove:
                    continue
                box_b = boxes_sorted[j]
                if _contains(box_a, box_b):
                    contained += 1
                    if contained >= min_contained:
                        to_remove.add(i)
                        _LOGGER.info(
                            f"Removing box {box_a} as it contains {contained} boxes."
                        )
                        break

        # Add boxes that are not removed
        for idx in range(n):
            if idx not in to_remove:
                filtered_bboxes.append(boxes_sorted[idx])
                filtered_labels.append(label)

    output_data = {"bboxes": filtered_bboxes, "labels": filtered_labels}

    return output_data
