import pytest
import numpy as np
from PIL import Image

from vision_agent_tools.shared_types import Device
from vision_agent_tools.models.sam2 import Sam2, Sam2Config


def test_sam2_point_segmentation_image(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)

    input_points = np.array([[110, 120]])
    input_label = np.array([1])
    input_box = None
    multimask_output = False

    response = shared_model(
        images=[test_image],
        input_points=input_points,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
    )

    assert len(response) == 1
    masks = response[0]["masks"]
    assert len(masks) == 1
    assert response[0]["scores"] == [0.9140625]

    for mask in masks:
        reverted_masks = rle_decode_array(mask)
        assert reverted_masks.shape == test_image.size[::-1]


def test_sam2_box_segmentation_image(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)

    input_points = None
    input_label = None
    input_box = np.array(
        [
            [60, 240, 380, 420],
            [65, 250, 150, 340],
        ]
    )
    multimask_output = False

    response = shared_model(
        images=[test_image],
        input_points=input_points,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
    )

    assert len(response) == 1
    masks = response[0]["masks"]
    assert len(masks) == 2
    assert response[0]["scores"] == [0.953125, 0.921875]

    for mask in masks:
        reverted_masks = rle_decode_array(mask)
        assert reverted_masks.shape == test_image.size[::-1]


def test_sam2_invalid_media(shared_model):
    with pytest.raises(ValueError):
        shared_model(images=["invalid media"])  # Invalid type for image

    with pytest.raises(ValueError):
        shared_model(images=[None])  # No image provided


def test_sam2_video_detection_segmentation(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    # Load a sample image and create a test video
    pil_test_image = Image.open(image_path)
    test_image = np.array(pil_test_image, dtype=np.uint8)
    test_video = np.array([test_image, np.zeros(test_image.shape, dtype=np.uint8)])

    input_points = np.array([[130, 170]])
    input_label = np.array([1])

    response = shared_model(
        video=test_video, input_points=input_points, input_label=input_label
    )

    assert len(response) == 2
    for frame_idx in range(len(response)): 
        masks = response[frame_idx]["masks"]
        assert len(masks) == 1
        assert response[frame_idx]["scores"] is None
        assert response[frame_idx]["logits"] is None

        for mask in masks:
            reverted_masks = rle_decode_array(mask)
            assert reverted_masks.shape == pil_test_image.size[::-1]


@pytest.fixture(scope="session")
def shared_model():
    return Sam2(
        model_config=Sam2Config(
            hf_model="facebook/sam2-hiera-large",
            device=Device.GPU,
        )
    )
