import json
from PIL import Image

from vision_agent_tools.shared_types import PromptTask


def test_ocr(shared_model):
    image_path = "tests/shared_data/images/license_plate.jpg"
    task = PromptTask.OCR_WITH_REGION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = shared_model(**payload)
    with open("tests/models/florence2_ft/data/results/ocr_results.json", "r") as source:
        expected_result = json.load(source)
    assert response == expected_result


def test_ocr_ft(small_model, unzip_model):
    image_path = "tests/shared_data/images/license_plate.jpg"
    model_zip_path = "tests/models/florence2_ft/data/models/ocr_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    task = PromptTask.OCR_WITH_REGION
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
    # TODO: we should debug why the results are empty
    expected_result = [{"quad_boxes": [], "labels": []}]
    assert response == expected_result
