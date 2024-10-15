from PIL import Image

from vision_agent_tools.shared_types import PromptTask


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


def test_caption_ft(small_model, unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.CAPTION
    model_zip_path = "tests/models/florence2_ft/data/models/caption_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    small_model.fine_tune(model_path)
    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = small_model(**payload)
    assert response == [{"text": "screw<loc_350><loc_670><loc_440><loc_770>"}]
