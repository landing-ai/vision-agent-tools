import pytest
from PIL import Image

from vision_agent_tools.models.florence2 import Florence2, Florence2Config
from vision_agent_tools.shared_types import PromptTask, Florence2ModelName


def test_more_detailed_caption(shared_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.MORE_DETAILED_CAPTION
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
    }
    response = shared_model(**payload)
    assert response == [
        {
            "text": "The image is of a pile of cereal. The cereal looks to be cheetos. "
            "The color of the cereal is yellow, orange, green and brown. There is a metal "
            "button in the middle of the pile. The background of the image is white."
        }
    ]


def test_more_detailed_caption_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.MORE_DETAILED_CAPTION
    model_zip_path = "tests/models/florence2/data/models/od_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    image = Image.open(image_path)

    config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    small_model = Florence2(config)
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
