from PIL import Image
from vision_agent_tools.tools.text_to_object_detection import TextToObjectDetection


def test_successful_text_to_object_detection():
    test_image = "car.jpg"

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    florencev2 = TextToObjectDetection()

    results = florencev2(image=image, task="<CAPTION>")
    caption = results["<CAPTION>"]

    assert caption == "A green car parked in front of a yellow building."
