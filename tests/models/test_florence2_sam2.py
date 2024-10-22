import json

import pytest
import numpy as np
from PIL import Image

from vision_agent_tools.shared_types import Florence2ModelName
from vision_agent_tools.models.florence2_sam2 import Florence2SAM2, Florence2SAM2Config


def test_florence2sam2_image(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)
    prompt = "tomato"

    response = shared_model(prompt, images=[test_image])

    with open(
        "tests/models/data/results/florence2sam2_image_results.json", "r"
    ) as dest:
        expected_results = json.load(dest)

    assert len(response) == len(expected_results)
    for result_frame, expected_result_frame in zip(response, expected_results):
        assert len(result_frame) == len(expected_result_frame)
        for result_annotation, expected_result_annotation in zip(
            result_frame, expected_result_frame
        ):
            assert result_annotation["id"] == expected_result_annotation["id"]
            assert result_annotation["bbox"] == expected_result_annotation["bbox"]
            assert (
                rle_decode_array(result_annotation["mask"]).shape
                == test_image.size[::-1]
            )
            assert result_annotation["label"] == expected_result_annotation["label"]


def test_florence2sam2_video(shared_model, rle_decode_array):
    tomatoes_image = Image.open("tests/shared_data/images/tomatoes.jpg")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    test_video = np.array([np_test_img, zeros])
    prompt = "tomato"

    response = shared_model(prompt, video=test_video)

    with open(
        "tests/models/data/results/florence2sam2_video_results.json", "r"
    ) as dest:
        expected_results = json.load(dest)

    assert len(response) == len(expected_results)
    for result_frame, expected_result_frame in zip(response, expected_results):
        assert len(result_frame) == len(expected_result_frame)
        for result_annotation, expected_result_annotation in zip(
            result_frame, expected_result_frame
        ):
            assert result_annotation["id"] == expected_result_annotation["id"]
            assert result_annotation["bbox"] == expected_result_annotation["bbox"]
            assert rle_decode_array(result_annotation["mask"]).shape == img_size[::-1]
            assert result_annotation["label"] == expected_result_annotation["label"]


def test_florence2sam2_invalid_media(shared_model):
    prompt = "tomato"
    with pytest.raises(ValueError):
        shared_model(prompt, images=["invalid media"])

    with pytest.raises(ValueError):
        shared_model(prompt, video="invalid media")

    with pytest.raises(ValueError):
        shared_model(prompt, video=np.array([1, 2, 3]))


@pytest.fixture(scope="module")
def shared_model():
    return Florence2SAM2(
        model_config=Florence2SAM2Config(
            hf_florence2_model=Florence2ModelName.FLORENCE_2_LARGE,
            hf_sam2_model="facebook/sam2-hiera-large",
        )
    )
