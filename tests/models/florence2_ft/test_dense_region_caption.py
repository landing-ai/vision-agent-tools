import pytest
from PIL import Image

from vision_agent_tools.shared_types import PromptTask


def test_dense_region_caption(shared_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.DENSE_REGION_CAPTION
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
    }
    response = shared_model(**payload)
    assert response == [{"labels": [], "bboxes": []}]


def test_dense_region_caption_ft(small_model, unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.DENSE_REGION_CAPTION
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
            == "The task DENSE_REGION_CAPTION is not supported yet if your are using "
            "a fine-tuned model."
        )
