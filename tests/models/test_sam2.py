import numpy as np
import pytest
from PIL import Image
from vision_agent_tools.models.sam2 import SAM2Model


def test_point_segmentation_sam2_image():
    """
    This test verifies that SAM2Model returns a valid response when passed an image.
    """
    test_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")

    sam2_model = SAM2Model()

    input_points = np.array([[110, 120]])
    input_label = np.array([1])
    input_box = None
    multimask_output = False

    masks, scores, logits = sam2_model.predict_image(
        image=test_image,
        input_points=input_points,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
    )

    assert len(masks) == 1
    assert len(scores) == 1
    assert len(logits) == 1

    for mask in masks:
        assert isinstance(mask, np.ndarray)
        assert mask.shape == test_image.size[::-1]


def test_box_segmentation_sam2_image():
    """
    This test verifies that SAM2Model returns a valid response when passed an image.
    """
    test_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")

    sam2_model = SAM2Model()

    input_points = None
    input_label = None
    input_box = np.array(
        [
            [60, 240, 380, 420],
            [65, 250, 150, 340],
        ]
    )
    multimask_output = False

    masks, scores, logits = sam2_model.predict_image(
        image=test_image,
        input_points=input_points,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
    )

    assert len(masks) == 2
    assert len(scores) == 2
    assert len(logits) == 2

    for mask in masks:
        assert isinstance(mask, np.ndarray)
        assert mask.shape[-2:] == test_image.size[::-1]


def test_sam2_invalid_media():
    """
    This test verifies that SAM2Model raises an error if the input media is not valid.
    """
    sam2_model = SAM2Model()

    with pytest.raises(ValueError):
        sam2_model.predict_image(image="invalid media")  # Invalid type for image

    with pytest.raises(ValueError):
        sam2_model.predict_image(image=None)  # No image provided


def test_sam2_image_no_prompts():
    """
    This test verifies that SAM2Model raises an error if neither points nor labels are provided.
    """
    test_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")
    sam2_model = SAM2Model()

    with pytest.raises(ValueError):
        sam2_model.predict_image(image=test_image, input_points=None, input_label=None)


def test_successful_video_detection_segmentation():
    """
    This test verifies that SAM2Model's predict_video method returns a valid response when passed a video.
    """
    # Load a sample image and create a test video
    test_image = np.array(
        Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB"), dtype=np.uint8
    )
    test_video = np.array([test_image, np.zeros(test_image.shape, dtype=np.uint8)])

    sam2_model = SAM2Model()

    input_points = np.array([[130, 170]])
    input_label = np.array([1])

    results = sam2_model.predict_video(
        video=test_video, input_points=input_points, input_label=input_label
    )

    assert len(results) == 2

    assert len(results[0]) == 1
    for obj_id, mask in results[0].items():
        assert isinstance(mask, np.ndarray)
        assert mask.shape[-2:] == test_image.shape[:2]

    assert len(results[1]) == 1
    for obj_id, mask in results[1].items():
        assert isinstance(mask, np.ndarray)
        assert mask.shape[-2:] == test_image.shape[:2]
