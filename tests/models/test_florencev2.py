from PIL import Image

from vision_agent_tools.models.florencev2 import Florencev2, PromptTask


def test_successful_florencev2_detection():
    test_image = "car.jpg"
    task = PromptTask.CAPTION

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    florencev2 = Florencev2()

    results = florencev2(images=image, task=task)
    caption = results[task]

    assert caption == "A green car parked in front of a yellow building."


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
