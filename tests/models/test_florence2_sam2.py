import json

import numpy as np
import pytest
from PIL import Image

from vision_agent_tools.models.florence2 import Florence2Config
from vision_agent_tools.models.florence2_sam2 import Florence2SAM2, Florence2SAM2Config
from vision_agent_tools.shared_types import Florence2ModelName


def test_florence2sam2_image(shared_model, assert_predictions):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)
    prompt = "tomato"

    response = shared_model(prompt, images=[test_image])

    with open(
        "tests/models/data/results/florence2sam2_image_tomato_results.json", "r"
    ) as dest:
        expected_results = json.load(dest)

    assert_predictions(response, expected_results, test_image.size[::-1])


def test_florence2sam2_image_ft(unzip_model, assert_predictions):
    image_path = "tests/shared_data/images/cereal.jpg"
    model_zip_path = (
        "tests/shared_data/models/caption_to_phrase_grounding_checkpoint.zip"
    )
    model_path = unzip_model(model_zip_path)
    prompt = "screw"
    image = Image.open(image_path)

    florence2_config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    config = Florence2SAM2Config(florence2_config=florence2_config)
    model = Florence2SAM2(config)
    response = model(prompt, images=[image])

    with open(
        "tests/models/data/results/florence2sam2_image_screw_ft_results.json", "r"
    ) as dest:
        expected_results = json.load(dest)

    assert_predictions(response, expected_results, image.size[::-1])


def test_florence2sam2_image_ft_base_ft(unzip_model, assert_predictions):
    image_path = "tests/shared_data/images/cereal.jpg"
    model_zip_path = (
        "tests/shared_data/models/caption_to_phrase_grounding_checkpoint.zip"
    )
    model_path = unzip_model(model_zip_path)
    prompt = "screw"
    image = Image.open(image_path)

    florence2_config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    config = Florence2SAM2Config(florence2_config=florence2_config)
    model = Florence2SAM2(config)
    response = model(prompt, images=[image])

    with open(
        "tests/models/data/results/florence2sam2_image_screw_ft_results.json", "r"
    ) as dest:
        expected_results_ft = json.load(dest)

    assert_predictions(response, expected_results_ft, image.size[::-1])

    # running prediction again without fine_tuning should reset the model to its base
    model.load_base()
    response = model(prompt, images=[image])

    assert response == [[]]

    # running prediction again with fine_tuning
    model.fine_tune(model_path)
    response = model(prompt, images=[image])

    assert_predictions(response, expected_results_ft, image.size[::-1])


def test_florence2sam2_video(shared_model, assert_predictions):
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

    assert_predictions(response, expected_results, img_size[::-1])


def test_florence2sam2_video_chunk_length(shared_model, assert_predictions):
    tomatoes_image = Image.open("tests/shared_data/images/tomatoes.jpg")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    test_video = np.array([np_test_img, zeros])
    prompt = "tomato"

    response = shared_model(prompt, video=test_video, chunk_length_frames=1)

    with open(
        "tests/models/data/results/florence2sam2_video_results.json", "r"
    ) as dest:
        expected_results = json.load(dest)

    assert_predictions(response, expected_results, img_size[::-1])


def test_florence2sam2_video_ft(unzip_model, assert_predictions):
    image = Image.open("tests/shared_data/images/cereal.jpg")
    img_size = image.size
    np_test_img = np.array(image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    test_video = np.array([np_test_img, zeros])
    prompt = "screw"
    model_zip_path = (
        "tests/shared_data/models/caption_to_phrase_grounding_checkpoint.zip"
    )
    model_path = unzip_model(model_zip_path)

    florence2_config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    config = Florence2SAM2Config(florence2_config=florence2_config)
    model = Florence2SAM2(config)
    response = model(prompt, video=test_video)

    with open(
        "tests/models/data/results/florence2sam2_video_ft_results.json", "r"
    ) as dest:
        expected_results = json.load(dest)

    assert_predictions(response, expected_results, img_size[::-1])


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
    return Florence2SAM2()


@pytest.fixture
def assert_predictions(rle_decode_array):
    def handler(
        response,
        expected_results,
        image_size,
        amount_of_matches: int = None,
        flex: int = 1,
    ):
        assert len(response) == len(expected_results)
        for result_frame, expected_result_frame in zip(response, expected_results):
            assert len(result_frame) == len(expected_result_frame)
            if amount_of_matches is None:
                amount_of_matches = len(expected_result_frame) - flex
            for result_annotation, expected_result_annotation in zip(
                result_frame, expected_result_frame
            ):
                assert result_annotation.keys() == expected_result_annotation.keys()
                assert result_annotation["id"] == expected_result_annotation["id"]
                assert np.allclose(
                    result_annotation["bbox"],
                    expected_result_annotation["bbox"],
                    rtol=1,
                    atol=1,
                )
                assert rle_decode_array(result_annotation["mask"]).shape == image_size
                assert result_annotation["label"] == expected_result_annotation["label"]

    return handler
