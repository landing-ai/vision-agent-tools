import numpy as np
from PIL import Image

from vision_agent_tools.tools.text_to_object_detection import (
    TextToObjectDetection,
    TextToObjectDetectionModel,
)
from vision_agent_tools.shared_types import Florence2ModelName
from vision_agent_tools.models.florence2 import Florence2Config


def test_text_to_object_detection_owlv2():
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = ["dog", "cat", "remote control"]

    tool = TextToObjectDetection(model=TextToObjectDetectionModel.OWLV2)
    response = tool(prompts, images=[image])

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


def test_text_to_object_detection_florence2():
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = ["dog", "cat", "remote control"]

    tool = TextToObjectDetection(model=TextToObjectDetectionModel.FLORENCE2)
    response = tool(prompts, images=[image])

    assert response == [
        {
            "labels": ["cat"],
            "bboxes": [
                [9.920000076293945, 53.03999710083008, 317.1199951171875, 474.0]
            ],
        }
    ]


def test_text_to_object_detection_florence2_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    image = Image.open(image_path)
    model_zip_path = (
        "tests/shared_data/models/caption_to_phrase_grounding_checkpoint.zip"
    )
    model_path = unzip_model(model_zip_path)
    prompts = ["screw"]

    model_config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    tool = TextToObjectDetection(TextToObjectDetectionModel.FLORENCE2, model_config)
    response = tool(prompts, images=[image])

    assert response == [
        {
            "bboxes": [
                [
                    723.968017578125,
                    1373.18408203125,
                    902.14404296875,
                    1577.984130859375,
                ]
            ],
            "labels": ["screw"],
        }
    ]


def test_text_to_object_detection_custom_confidence():
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = ["dog", "cat", "remote control"]

    tool = TextToObjectDetection(model=TextToObjectDetectionModel.OWLV2)
    response = tool(prompts, images=[image], confidence=0.7)
    assert response == [{"scores": [], "labels": [], "bboxes": []}]


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
