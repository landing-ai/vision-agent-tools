from PIL import Image

from vision_agent_tools.tools.florencev2 import Florencev2, PromptTask


def test_successful_florencev2_detection():
    test_image = "car.jpg"
    task = PromptTask.CAPTION

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    florencev2 = Florencev2()

    results = florencev2(image=image, task=task)
    caption = results[task]

    assert caption == "A green car parked in front of a yellow building."
