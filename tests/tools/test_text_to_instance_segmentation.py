import json

import pytest
import numpy as np
from PIL import Image

from vision_agent_tools.tools.text_to_instance_segmentation import (
    TextToInstanceSegmentationTool,
    TextToInstanceSegmentationModel,
)


def test_text_to_instance_segmentation_image(shared_tool, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)
    prompt = "tomato"

    response = shared_tool(prompt, images=[test_image])

    with open("tests/models/data/florence2sam2_image_results.json", "r") as dest:
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


def test_text_to_instance_segmentation_video(shared_tool, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)
    img_size = test_image.size
    np_test_img = np.array(test_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    test_video = np.array([np_test_img, zeros])
    prompt = "tomato"

    response = shared_tool(prompt, video=test_video)

    with open("tests/models/data/florence2sam2_video_results.json", "r") as dest:
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


def test_text_to_instance_segmentation_invalid_media(shared_tool):
    prompt = "tomato"

    with pytest.raises(ValueError):
        shared_tool(prompt, images=["invalid media"])

    with pytest.raises(ValueError):
        shared_tool(prompt, video="invalid media")

    with pytest.raises(ValueError):
        shared_tool(prompt, video=np.array([1, 2, 3]))


@pytest.fixture(scope="module")
def shared_tool():
    return TextToInstanceSegmentationTool(
        model=TextToInstanceSegmentationModel.FLORENCE2SAM2
    )
