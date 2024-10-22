from PIL import Image

from vision_agent_tools.models.florence2_qa import FlorenceQA


def test_florence2_qa():
    test_image = "car.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    florence_qa = FlorenceQA()

    answer = florence_qa(image=image, question="what is the color of the car?")
    assert answer == {"text": "turquoise"}
