import pytest
import numpy as np
from PIL import Image

from vision_agent_tools.tools.text_to_object_detection import (
    TextToObjectDetection,
    TextToObjectDetectionModel,
)
from vision_agent_tools.models.owlv2 import OWLV2Config


def test_text_to_object_detection_owlv2(shared_tool_owlv2):
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = ["dog", "cat", "remote control"]

    response = shared_tool_owlv2(prompts, images=[image])

    expected_response = [
        {
            "scores": [
                0.6933691501617432,
                0.6252092719078064,
                0.6656279563903809,
                0.6483550071716309,
            ],
            "labels": ["remote control", "remote control", "cat", "cat"],
            "bboxes": [
                [41.71875, 72.65625, 173.75, 117.03125],
                [334.0625, 78.3203125, 370.3125, 190.0],
                [340.0, 24.6875, 639.375, 369.375],
                [11.5625, 55.625, 315.625, 471.5625],
            ],
        }
    ]
    check_results(response, expected_response)


def test_text_to_object_detection_florence2(shared_tool_florence2):
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = ["dog", "cat", "remote control"]

    response = shared_tool_florence2(prompts, images=[image])

    assert response == [
        {
            "labels": ["cat"],
            "bboxes": [
                [9.920000076293945, 53.03999710083008, 317.1199951171875, 474.0]
            ],
        }
    ]


def test_text_to_object_detection_custom_confidence():
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = ["dog", "cat", "remote control"]

    tool = TextToObjectDetection(
        model=TextToObjectDetectionModel.OWLV2, model_config=OWLV2Config(confidence=0.7)
    )
    response = tool(prompts, images=[image])
    assert response == [{"scores": [], "labels": [], "bboxes": []}]


@pytest.fixture(scope="module")
def shared_tool_owlv2():
    return TextToObjectDetection(model=TextToObjectDetectionModel.OWLV2)


@pytest.fixture(scope="module")
def shared_tool_florence2():
    return TextToObjectDetection(model=TextToObjectDetectionModel.FLORENCE2)


def check_results(
    response, expected_response, amount_of_matches: int = None, flex: int = 1
):
    # sort the results by score to make the comparison easier
    response = sorted(response, key=lambda x: x["scores"], reverse=True)
    expected_response = sorted(
        expected_response, key=lambda x: x["scores"], reverse=True
    )

    for item, expected_result in zip(response, expected_response):
        if amount_of_matches is None:
            amount_of_matches = len(expected_result["bboxes"]) - flex

        assert (
            abs(len(item["bboxes"]) - len(expected_result["bboxes"]))
            <= amount_of_matches
        )
        for bbox, expected_bbox in zip(item["bboxes"], expected_result["bboxes"]):
            assert np.allclose(bbox, expected_bbox, rtol=1, atol=1)

        assert (
            abs(len(item["labels"]) - len(expected_result["labels"]))
            <= amount_of_matches
        )
        count_equal_labels = 0
        for lab, expected_lab in zip(item["labels"], expected_result["labels"]):
            if lab == expected_lab:
                count_equal_labels += 1
        assert (
            abs(count_equal_labels - len(item["labels"])) <= amount_of_matches
        ), f"{item['labels']}, {expected_result['labels']}"

        assert (
            abs(len(item["scores"]) - len(expected_result["scores"]))
            <= amount_of_matches
        )
        for bbox, expected_bbox in zip(item["scores"], expected_result["scores"]):
            np.testing.assert_almost_equal(bbox, expected_bbox, decimal=1)
