import pytest
from PIL import Image

from vision_agent_tools.shared_types import PromptTask


def test_more_detailed_caption(small_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.MORE_DETAILED_CAPTION
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
    }
    response = small_model(**payload)
    assert response == [
        {
            "text": "The image is of a pile of cereal. The cereal looks to be cheetos. "
            "The color of the cereal is yellow, orange, green and brown. There is a metal "
            "button in the middle of the pile. The background of the image is white."
        }
    ]


def test_more_detailed_caption_ft(small_model, unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.MORE_DETAILED_CAPTION
    model_zip_path = "tests/models/florence2_ft/data/models/od_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    image = Image.open(image_path)

    small_model.fine_tune(model_path)
    payload = {
        "images": [image],
        "task": task,
    }
    with pytest.raises(ValueError) as exc:
        small_model(**payload)
        assert (
            exc.value
            == "The task MORE_DETAILED_CAPTION is not supported yet if your are using a fine-tuned model."
        )