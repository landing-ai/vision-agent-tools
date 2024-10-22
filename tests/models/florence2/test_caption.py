from PIL import Image

from vision_agent_tools.models.florence2 import Florence2, Florence2Config
from vision_agent_tools.shared_types import PromptTask, Florence2ModelName


def test_caption_cereal(shared_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.CAPTION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = shared_model(**payload)
    assert response == [{"text": "A pile of colorful doughnuts sitting on a table."}]


def test_caption_car(shared_model):
    image_path = "tests/shared_data/images/car.jpg"
    task = PromptTask.CAPTION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = shared_model(**payload)
    assert response == [{"text": "A green car parked in front of a yellow building."}]


def test_caption_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.CAPTION
    model_zip_path = "tests/models/florence2/data/models/caption_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    small_model = Florence2(config)
    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = small_model(**payload)
    assert response == [{"text": "screw<loc_350><loc_670><loc_440><loc_770>"}]
