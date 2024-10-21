import json

import pytest
import numpy as np
from PIL import Image

from vision_agent_tools.shared_types import Florence2ModelName, Device
from vision_agent_tools.models.florence2_sam2 import Florence2SAM2, Florence2SAM2Config


def test_florence2sam2_image(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)
    prompt = "tomato"

    response = shared_model(prompt, images=[test_image])

    with open("tests/models/data/florence2sam2_image_results.json", "r") as dest:
        expected_results = json.load(dest)

    assert expected_results == response
    reverted_masks = rle_decode_array(response[0][0]["mask"])
    assert reverted_masks.shape == test_image.size[::-1]


def test_florence2sam2_video(shared_model, rle_decode_array):
    tomatoes_image = Image.open("tests/shared_data/images/tomatoes.jpg")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    test_video = np.array([np_test_img, zeros])
    prompt = "tomato"

    response = shared_model(prompt, video=test_video)

    with open("tests/models/data/florence2sam2_video_results.json", "r") as dest:
        expected_results = json.load(dest)

    assert expected_results == response
    # only check the first frame since the second frame is all zeros
    reverted_masks =  rle_decode_array(response[0][0]["mask"])
    assert reverted_masks.shape == img_size[::-1]


def test_florence2sam2_invalid_media(shared_model):
    prompt = "tomato"
    with pytest.raises(ValueError):
        shared_model(prompt, images=["invalid media"])

    with pytest.raises(ValueError):
        shared_model(prompt, video="invalid media")

    with pytest.raises(ValueError):
        shared_model(prompt, video=np.array([1, 2, 3]))


@pytest.fixture(scope="session")
def shared_model():
    return Florence2SAM2(
        model_config=Florence2SAM2Config(
            hf_florence2_model=Florence2ModelName.FLORENCE_2_LARGE,
            hf_sam2_model="facebook/sam2-hiera-large",
            device=Device.GPU,
        )
    )
