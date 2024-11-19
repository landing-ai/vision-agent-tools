import json

import pytest
import numpy as np
from PIL import Image

from vision_agent_tools.models.owlv2 import Owlv2


def test_owlv2_image(shared_model):
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = ["dog", "cat", "remote control"]

    response = shared_model(prompts, images=[image])

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


def test_owlv2_removing_extra_bbox(shared_model):
    image_path = "tests/shared_data/images/eggs-food-easter-food-drink-44c10e-1024.jpg"
    image = Image.open(image_path)
    prompts = ["egg"]

    response = shared_model(prompts, images=[image])

    assert len(response) == 1
    item = response[0]
    assert len(item["bboxes"]) == 40
    assert len([label == "egg" for label in item["labels"]]) == 40


def test_owlv2_image_with_nms(shared_model):
    image_path = "tests/shared_data/images/surfers_with_shark.png"
    image = Image.open(image_path)
    prompts = ["surfer", "shark"]

    response = shared_model(prompts, images=[image], confidence=0.2, nms_threshold=1.0)

    expected_response = [
        {
            "scores": [
                0.6650843620300293,
                0.28987398743629456,
                0.5511208176612854,
                0.420064240694046,
            ],
            "labels": ["shark", "surfer", "surfer", "surfer"],
            "bboxes": [
                [
                    118.4044189453125,
                    166.135986328125,
                    281.32177734375,
                    238.452392578125,
                ],
                [338.84619140625, 129.703857421875, 385.96142578125, 217.086181640625],
                [340.2158203125, 142.030517578125, 388.974609375, 199.281005859375],
                [165.451171875, 282.41748046875, 203.80078125, 359.11669921875],
            ],
        }
    ]
    check_results(response, expected_response)

    response = shared_model(prompts, images=[image], confidence=0.2, nms_threshold=0.3)

    expected_response = [
        {
            "scores": [0.6650843620300293, 0.5511208176612854, 0.420064240694046],
            "labels": ["shark", "surfer", "surfer"],
            "bboxes": [
                [
                    118.4044189453125,
                    166.135986328125,
                    281.32177734375,
                    238.452392578125,
                ],
                [340.2158203125, 142.030517578125, 388.974609375, 199.281005859375],
                [165.451171875, 282.41748046875, 203.80078125, 359.11669921875],
            ],
        }
    ]
    check_results(response, expected_response)


def test_owlv2_image_with_large_prompt(shared_model):
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompts = [
        """
        A photo of a cat that is sleeping next to a remote control. The cat has a
        light brown color with black spots and seems to be wearing a light green
        necklace. It also seems to be stretching its right leg and next to its
        left leg it is stepping on the tail
    """
    ]

    response = shared_model(prompts, images=[image])

    expected_response = [
        {
            "scores": [
                0.35723111033439636,
                0.3886180520057678,
                0.18877223134040833,
                0.22185605764389038,
            ],
            "labels": [prompts[0], prompts[0], prompts[0], prompts[0]],
            "bboxes": [
                [41.71875, 72.65625, 173.75, 117.03125],
                [334.0625, 78.3203125, 370.3125, 190.0],
                [340.0, 24.6875, 639.375, 369.375],
                [11.5625, 55.625, 315.625, 471.5625],
            ],
        }
    ]
    check_results(response, expected_response)


def test_owlv2_video(shared_model, bytes_to_np):
    test_video = "tests/shared_data/videos/test_video_5_frames.mp4"
    prompts = ["car"]
    with open(test_video, "rb") as f:
        video_bytes = f.read()
        video = bytes_to_np(video_bytes)

    response = shared_model(prompts, video=video)
    with open("tests/models/data/results/owlv2_video_results.json", "r") as dest:
        expected_results = json.load(dest)

    check_results(response, expected_results)


@pytest.fixture(scope="module")
def shared_model():
    return Owlv2()


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
