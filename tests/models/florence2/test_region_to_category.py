import pytest
from PIL import Image

from vision_agent_tools.models.florence2 import Florence2, Florence2Config
from vision_agent_tools.shared_types import PromptTask, Florence2ModelName


def test_region_to_category(shared_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.REGION_TO_CATEGORY
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
    }
    response = shared_model(**payload)
    # TODO: change to bbox output, but needs to parse florence2 bbox output
    assert response == [{"text": "doughnut<loc_0><loc_0><loc_998><loc_998>"}]


def test_region_to_category_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.REGION_TO_CATEGORY
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
            exc.value == "The task REGION_TO_CATEGORY is not supported yet if your are "
            "using a fine-tuned model."
        )
