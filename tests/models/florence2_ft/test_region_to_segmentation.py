import pytest
from PIL import Image

from vision_agent_tools.shared_types import PromptTask


def test_region_to_segmentation(shared_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.REGION_TO_SEGMENTATION
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
    }
    response = shared_model(**payload)
    assert response == [
        {
            "labels": [""],
            "polygons": [
                [
                    [
                        1.0240000486373901,
                        1.0240000486373901,
                        2046.97607421875,
                        2042.880126953125,
                        1.0240000486373901,
                        2046.97607421875,
                    ]
                ]
            ],
        }
    ]


def test_region_to_segmentation_ft(small_model, unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.REGION_TO_SEGMENTATION
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
            == "The task REGION_TO_SEGMENTATION is not supported yet if your are "
            "using a fine-tuned model."
        )
