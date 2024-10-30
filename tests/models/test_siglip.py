import pytest
from PIL import Image

from vision_agent_tools.models.siglip import Siglip, SiglipTask


def test_zero_shot_image_classification(model):
    image = Image.open("tests/shared_data/images/car.jpg")
    candidate_labels = ["2 cars", "1 car", "1 airplane", "1 boat"]

    result = model(
        image=image,
        candidate_labels=candidate_labels,
        task=SiglipTask.ZERO_SHOT_IMG_CLASSIFICATION,
    )

    assert result is not None
    assert len(result) == 4
    assert all("score" in item for item in result)
    assert all("label" in item for item in result)
    assert all(isinstance(item["score"], float) for item in result)
    assert all(isinstance(item["label"], str) for item in result)


def test_zero_shot_image_classification_with_invalid_task(model):
    image = Image.open("tests/shared_data/images/car.jpg")
    candidate_labels = ["2 cars", "1 car", "1 airplane", "1 boat"]

    with pytest.raises(ValueError):
        model(
            image=image,
            candidate_labels=candidate_labels,
            task="invalid_task",
        )


def test_zero_shot_image_classification_correctness(model):
    image = Image.open("tests/shared_data/images/car.jpg")
    candidate_labels = ["a car", "an airplane", "one boat"]

    result = model(
        image=image,
        candidate_labels=candidate_labels,
        task=SiglipTask.ZERO_SHOT_IMG_CLASSIFICATION,
    )

    assert result is not None
    assert len(result) == 3
    assert result[0]["score"] > result[1]["score"]
    assert result[0]["score"] > result[2]["score"]


@pytest.fixture(scope="module")
def model():
    return Siglip()
