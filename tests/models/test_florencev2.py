import pytest
from PIL import Image
from pydantic import ValidationError

from vision_agent_tools.models.florencev2 import Florencev2, PromptTask


def test_successful_florencev2_detection():
    test_image = "car.jpg"
    task = PromptTask.CAPTION

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    florencev2 = Florencev2()

    results = florencev2(image=image, task=task)
    caption = results[task]

    assert caption == "A green car parked in front of a yellow building."


def test_successful_florencev2_od_detection_with_nms():
    test_image = "car.jpg"
    task = PromptTask.CAPTION_TO_PHRASE_GROUNDING

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    florencev2 = Florencev2()

    results = florencev2(image=image, task=task, prompt="car", nms_threshold=0.1)
    prediction = results[task]

    assert len(prediction["labels"]) == 1
    assert len(prediction["bboxes"]) == 1
    assert len(prediction["bboxes"][0]) == 4


def test_successful_florencev2_detection_video(random_video_generator):
    video_np = random_video_generator()
    task = PromptTask.CAPTION
    florencev2 = Florencev2()

    results = florencev2(video=video_np, task=task)
    captions = results[0][task]

    assert len(captions) > 0


def test_batch_ocr():
    ocr_image_1 = Image.open("tests/models/data/florencev2/ocr_image_1.jpg")
    ocr_image_2 = Image.open("tests/models/data/florencev2/ocr_image_2.jpg")
    ocr_image_3 = Image.open("tests/models/data/florencev2/ocr_image_3.jpg")
    task = PromptTask.OCR

    florencev2 = Florencev2()

    results = florencev2(images=[ocr_image_1, ocr_image_2, ocr_image_3], task=task)

    assert len(results) == 3

    assert results[0][task] == "HOWARD JACOBSONREDBACK"
    assert results[1][task] == "Panasync E70"
    assert results[2][task] == "conditions"


def test_batch_size_validation():
    model = Florencev2()
    task = PromptTask.OCR
    image = Image.new("RGB", (224, 224))

    # List of invalid batch_size values (less than 1 and greater than 10)
    invalid_batch_sizes = [0, -1, 11, 15]

    for invalid_batch_size in invalid_batch_sizes:
        with pytest.raises(ValidationError):
            res = model(task=task, image=image, batch_size=invalid_batch_size)
            print(res)
