import pytest
from PIL import Image

from vision_agent_tools.shared_types import PromptTask


def test_region_to_description(small_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.REGION_TO_DESCRIPTION
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
    }
    response = small_model(**payload)
    assert response == [{"text": "doughnuts in various colors on a white surface"}]


def test_region_to_description_ft(small_model, unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.REGION_TO_DESCRIPTION
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
            exc.value == "The task REGION_TO_DESCRIPTION is not supported yet if "
            "your are using a fine-tuned model."
        )
