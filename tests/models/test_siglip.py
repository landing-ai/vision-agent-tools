import pytest
from PIL import Image

from vision_agent_tools.models.siglip import Siglip, SiglipTask


def test_zero_shot_image_classification(model):
    image = Image.open("tests/shared_data/images/car.jpg")
    candidate_labels = ["2 cars", "1 car", "1 airplane", "1 boat"]

    result = model(
        image=image,
        candidate_labels=candidate_labels,
        task=SiglipTask.ZERO_SHOT_IMAGE_CLASSIFICATION,
    )

    assert result is not None
    assert len(result["labels"]) == 4
    assert len(result["scores"]) == 4
    assert all(isinstance(label, str) for label in result["labels"])
    assert all(isinstance(score, float) for score in result["scores"])


def test_zero_shot_image_classification_correctness(model):
    image = Image.open("tests/shared_data/images/car.jpg")
    candidate_labels = ["a car", "an airplane", "one boat"]

    result = model(
        image=image,
        candidate_labels=candidate_labels,
        task=SiglipTask.ZERO_SHOT_IMAGE_CLASSIFICATION,
    )

    assert result is not None
    assert len(result["labels"]) == 3
    assert len(result["scores"]) == 3
    assert result["scores"][0] > result["scores"][1]
    assert result["scores"][0] > result["scores"][2]


def test_zero_shot_image_classification_no_image(model):
    candidate_labels = ["2 cars", "1 car", "1 airplane", "1 boat"]

    with pytest.raises(ValueError):
        model(
            candidate_labels=candidate_labels,
            task=SiglipTask.ZERO_SHOT_IMAGE_CLASSIFICATION,
        )


def test_zero_shot_image_classification_with_empty_candidate_labels(model):
    image = Image.open("tests/shared_data/images/car.jpg")
    candidate_labels = []

    with pytest.raises(ValueError):
        model(
            image=image,
            candidate_labels=candidate_labels,
            task=SiglipTask.ZERO_SHOT_IMAGE_CLASSIFICATION,
        )


def test_zero_shot_image_classification_with_invalid_task(model):
    image = Image.open("tests/shared_data/images/car.jpg")
    candidate_labels = ["2 cars", "1 car", "1 airplane", "1 boat"]

    with pytest.raises(ValueError):
        model(
            image=image,
            candidate_labels=candidate_labels,
            task="invalid_task",
        )


@pytest.fixture(scope="module")
def model():
    return Siglip()
