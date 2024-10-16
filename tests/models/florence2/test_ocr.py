import pytest
from PIL import Image

from vision_agent_tools.shared_types import PromptTask


def test_ocr_batch(shared_model):
    ocr_image_1 = Image.open("tests/shared_data/images/ocr_image_1.jpg")
    ocr_image_2 = Image.open("tests/shared_data/images/ocr_image_2.jpg")
    ocr_image_3 = Image.open("tests/shared_data/images/ocr_image_3.jpg")
    task = PromptTask.OCR

    payload = {
        "images": [ocr_image_1, ocr_image_2, ocr_image_3],
        "task": task,
    }
    response = shared_model(**payload)

    assert len(response) == 3
    assert response == [
        {"text": "HOWARD JACOBSONREDBACK"},
        {"text": "PanaspireE70"},
        {"text": "conditions"},
    ]


def test_ocr_ft(small_model, unzip_model):
    image_path = "tests/shared_data/images/ocr_image_1.jpg"
    model_zip_path = "tests/models/florence2/data/models/od_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    task = PromptTask.OCR
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
            == "The task OCR is not supported yet if your are using a fine-tuned model."
        )
