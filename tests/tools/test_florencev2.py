from PIL import Image

from vision_agent_tools.tools.florencev2 import Florencev2


def test_successful_florencev2_detection():
    test_image = "car.jpg"
    prompt = "<OD>"

    image = Image.open(f"tests/tools/data/owlv2/{test_image}")

    florencev2 = Florencev2()

    results = florencev2(image, prompt=prompt)

    assert len(results) > 0
