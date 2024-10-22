import json
from PIL import Image

from vision_agent_tools.models.florence2 import Florence2, Florence2Config
from vision_agent_tools.shared_types import PromptTask, Florence2ModelName


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
    with open("tests/models/florence2/data/results/ocr_results.json", "r") as source:
        expected_result = json.load(source)
    assert response == expected_result


def test_ocr_ft(unzip_model):
    image_path = "tests/shared_data/images/license_plate.jpg"
    model_zip_path = "tests/models/florence2/data/models/ocr_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    task = PromptTask.OCR_WITH_REGION
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
    # TODO: we should debug why the results are empty
    expected_result = [{"quad_boxes": [], "labels": []}]
    assert response == expected_result
