import numpy as np
import pytest
from PIL import Image
from vision_agent_tools.models.sam2 import SAM2Model


def test_successful_sam2_image():
    """
    This test verifies that SAM2Model returns a valid response when passed an image.
    """
    test_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")

    sam2_model = SAM2Model()

    input_points = np.array([[10, 10], [20, 20]])
    input_label = np.array([1, 1])
    input_box = None
    multimask_output = False

    masks, scores, logits = sam2_model.predict_image(
        image=test_image,
        input_points=input_points,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
    )

    assert len(masks) > 0  # At least one mask should be returned
    assert len(scores) > 0  # At least one score should be returned
    assert len(logits) > 0  # At least one logits should be returned

    for mask in masks:
        assert isinstance(mask, np.ndarray)
        assert mask.shape == test_image.size[::-1]


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
