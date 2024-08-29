from PIL import Image

from vision_agent_tools.models.florencev2 import Florencev2, PromptTask


def test_successful_florencev2_detection():
    test_image = "car.jpg"
    task = PromptTask.CAPTION

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    florencev2 = Florencev2()

    results = florencev2(image=image, task=task)
    caption = results[task]

    assert caption == "A green car parked in front of a yellow building."


def test_successful_florencev2_detection_video(random_video_generator):
    video_np = random_video_generator()
    task = PromptTask.CAPTION

    florencev2 = Florencev2()

    results = florencev2(video=video_np, task=task)
    captions = results[task]

    assert len(captions) > 0
