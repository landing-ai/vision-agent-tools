import logging
from typing import Any

from vision_agent_tools.models.utils import calculate_bbox_iou

_LOGGER = logging.getLogger(__name__)
_AREA_THRESHOLD = 0.82


def filter_bbox_predictions(
    predictions: dict[str, Any],
    image_size: tuple[int, int],
    *,
    nms_threshold: float = 0.3,
    bboxes_key: str = "bboxes",
    label_key: str = "labels",
) -> dict[str, Any]:
    new_preds = {}

    # Remove invalid bboxes, other filters rely on well formed bboxes
    bboxes_to_remove = _filter_invalid_bboxes(
        predictions=predictions,
        image_size=image_size,
        bboxes_key=bboxes_key,
    )

    new_preds = _remove_bboxes(predictions, bboxes_to_remove)

    # Remove the whole image bounding box if it is predicted
    bboxes_to_remove = _remove_whole_image_bbox(new_preds, image_size, bboxes_key)
    new_preds = _remove_bboxes(new_preds, bboxes_to_remove)

    # Apply a dummy agnostic Non-Maximum Suppression (NMS) to get rid of any
    # overlapping predictions on the same object
    bboxes_to_remove = _dummy_agnostic_nms(new_preds, nms_threshold, bboxes_key)
    new_preds = _remove_bboxes(new_preds, bboxes_to_remove)

    # Remove redundant boxes (boxes that are completely covered by another box)
    bboxes_to_remove = _filter_redundant_boxes(
        new_preds[bboxes_key], new_preds[label_key]
    )
    new_preds = _remove_bboxes(new_preds, bboxes_to_remove)

    return new_preds


def _remove_whole_image_bbox(
    predictions: dict[str, Any], image_size: tuple[int, int], bboxes_key: str = "bboxes"
) -> list[int]:
    # TODO: remove polygons that covers the whole image
    bboxes_to_remove = []
    img_area = image_size[0] * image_size[1]
    for idx, bbox in enumerate(predictions[bboxes_key]):
        x1, y1, x2, y2 = bbox
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / img_area > _AREA_THRESHOLD:
            _LOGGER.warning(
                "Model predicted the whole image bounding box, therefore we are "
                f"removing this prediction: {bbox}, image size: {image_size}."
            )
            bboxes_to_remove.append(idx)
    return bboxes_to_remove


def _remove_bboxes(
    predictions: dict[str, Any], bboxes_to_remove: list[int]
) -> dict[str, Any]:
    new_preds = {}
    for key, value in predictions.items():
        new_preds[key] = [
            value[idx] for idx in range(len(value)) if idx not in bboxes_to_remove
        ]
    return new_preds


def _dummy_agnostic_nms(
    predictions: dict[str, Any], nms_threshold: float, bboxes_key: str = "bboxes"
) -> list[int]:
    """
    Applies a dummy agnostic Non-Maximum Suppression (NMS) to filter overlapping predictions.

    Parameters:
        predictions:
            All predictions, including bboxes and labels.
        nms_threshold:
            The IoU threshold value used for NMS.

    Returns:
        list[int]:
            Indexes to remove from the predictions.
    """
    bboxes_to_keep = []
    prediction_items = {idx: pred for idx, pred in enumerate(predictions[bboxes_key])}

    while prediction_items:
        # the best prediction here is the first prediction since florence2 don't
        # have score per prediction
        best_prediction_idx = next(iter(prediction_items))
        best_prediction_bbox = prediction_items[best_prediction_idx]
        bboxes_to_keep.append(best_prediction_idx)

        new_prediction_items = {}
        for idx, pred in prediction_items.items():
            if calculate_bbox_iou(best_prediction_bbox, pred) < nms_threshold:
                bboxes_to_keep.append(idx)
                new_prediction_items[idx] = pred
        prediction_items = new_prediction_items

    bboxes_to_remove = []
    for idx, bbox in enumerate(predictions[bboxes_key]):
        if idx not in bboxes_to_keep:
            _LOGGER.warning(
                "Model predicted overlapping bounding boxes, therefore we are "
                f"removing this prediction: {bbox}."
            )
            bboxes_to_remove.append(idx)

    return bboxes_to_remove


def _filter_redundant_boxes(
    bboxes: list[list[float]], labels: list[str], min_contained: int = 2
) -> list[int]:
    """Filters out redundant bounding boxes that fully contain multiple smaller
    boxes of the same label.

    Parameters:
        bboxes:
            List of bounding boxes.
        labels:
            List of bounding labels.
        min_contained:
            Minimum number of contained boxes to consider a box redundant.

    Returns:
        list[int]:
            Indexes to remove from the bboxes.
    """
    bboxes_to_remove = []

    # Organize boxes by label and idx
    label_to_boxes = {}
    for idx, bbox, label in zip(range(len(bboxes)), bboxes, labels):
        label_to_boxes.setdefault(label, []).append({"bbox": bbox, "idx": idx})

    for label, boxes_and_idx in label_to_boxes.items():
        n = len(boxes_and_idx)
        if n < min_contained + 1:
            # Not enough boxes to have redundancies
            continue

        # Sort boxes by area descending
        boxes_and_idx_sorted = sorted(
            boxes_and_idx,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            reverse=True,
        )

        to_remove = set()
        for i in range(n):
            if i in to_remove:
                continue
            box_a = boxes_and_idx_sorted[i]["bbox"]
            contained = 0
            for j in range(n):
                if i == j or j in to_remove:
                    continue
                box_b = boxes_and_idx_sorted[j]["bbox"]
                if _contains(box_a, box_b):
                    contained += 1
                    if contained >= min_contained:
                        to_remove.add(i)
                        _LOGGER.info(
                            f"Removing box {box_a} as it contains {contained} boxes."
                        )
                        bboxes_to_remove.append(boxes_and_idx_sorted[i]["idx"])
                        break

    return bboxes_to_remove


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


def _filter_invalid_bboxes(
    predictions: dict[str, Any],
    image_size: tuple[int, int],
    bboxes_key: str = "bboxes",
) -> list[int]:
    """Filters out invalid bounding boxes from the given predictions and
    returns a list of indices of invalid boxes.

    Args:
        predictions: A dictionary containing 'bboxes' and 'labels' keys.
        image_size: A tuple representing the image width and height.
        bboxes_key: The key for bounding boxes in the predictions dictionary.

    Returns:
        A list of indices of invalid bounding boxes.
    """
    width, height = image_size

    invalid_indices = []

    for idx, bbox in enumerate(predictions[bboxes_key]):
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            invalid_indices.append(idx)
            _LOGGER.warning(f"Removing invalid bbox {bbox}")

    return invalid_indices
