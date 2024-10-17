import json
from typing import Any

import pytest
import numpy as np
from PIL import Image

from vision_agent_tools.shared_types import Florence2ModelName, Device
from vision_agent_tools.models.florence2_sam2 import Florence2SAM2, Florence2SAM2Config


def test_florence2sam2_image(shared_model):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)
    prompt = "tomato"

    response = shared_model(prompt, images=[test_image])

    with open("tests/models/data/florence2sam2_image_results.json", "r") as dest:
        expected_results = json.load(dest)

    assert expected_results == response
    for idx in range(len(response[0]["masks"])):
        reverted_masks = _rle_to_binary_mask(response[0]["masks"][idx])
        assert reverted_masks.shape == test_image.size[::-1]


def test_florence2sam2_video(shared_model):
    tomatoes_image = Image.open("tests/shared_data/images/tomatoes.jpg").convert("RGB")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    test_video = np.array([np_test_img, zeros])

    results = shared_model(prompts=["tomato"], video=test_video)

    # The list should have 2 keys for the two frames in the video
    assert len(results) == 2
    # The first frame should have 22 instances of the tomato class
    assert len(results[0]) == 22
    # The second frame should not have any tomato class since it is all zeros
    assert len(results[1]) == 0
    # First frame
    for instance in results[0]:
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == np_test_img.shape[:2]
        assert instance.label == "tomato"


def test_nms_florence2sam2_video(shared_model):
    tomatoes_image = Image.open("tests/shared_data/images/tomatoes.jpg").convert("RGB")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    test_video = np.array([np_test_img, zeros])

    results = shared_model(prompts=["tomato"], video=test_video, nms_threshold=0.1)

    # The list should have 2 keys for the two frames in the video
    assert len(results) == 2
    # The first frame should have 22 instances of the tomato class
    assert len(results[0]) == 22
    # The second frame should not have any tomato class since it is all zeros
    assert len(results[1]) == 0
    # First frame
    for instance in results[0]:
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == np_test_img.shape[:2]
        assert instance.label == "tomato"


def test_florence2sam2_invalid_media(shared_model):
    with pytest.raises(ValueError):
        shared_model(prompts=["tomatoe"], image="invalid media")

    with pytest.raises(ValueError):
        shared_model(prompts=["tomato"], video="invalid media")

    with pytest.raises(AssertionError):
        shared_model(prompts=["tomatoe"], video=np.array([1, 2, 3]))


@pytest.fixture(scope="session")
def shared_model():
    return Florence2SAM2(
        model_config=Florence2SAM2Config(
            hf_florence2_model=Florence2ModelName.FLORENCE_2_LARGE,
            hf_sam2_model="facebook/sam2-hiera-large",
            device=Device.GPU,
        )
    )


def _rle_to_binary_mask(rle: dict[str, Any]) -> np.ndarray:
    # Convert the counts to a flat array of 0s and 1s
    flat_mask = np.zeros(sum(rle["counts"]), dtype=np.uint8)
    current_value = 1
    position = 0
    for count in rle["counts"]:
        flat_mask[position : position + count] = current_value
        # Flip between 1 and 0
        current_value = 1 - current_value
        position += count

    # Reshape the flat array back to the original mask shape
    binary_mask = flat_mask.reshape(rle["size"], order="F").astype(np.uint8)
    return binary_mask[0]
