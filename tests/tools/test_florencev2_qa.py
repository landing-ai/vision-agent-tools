from PIL import Image

from vision_agent_tools.tools.florencev2_qa import FlorenceQA


def test_successful_florencev2_qa():
    test_image = "car.jpg"

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    florence_qa = FlorenceQA()

    answer = florence_qa(image=image, question="what is the color of the car?")

    assert answer == "green"
