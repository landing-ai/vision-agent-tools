from PIL import Image

from vision_agent_tools.models.owlv2 import Owlv2


def test_successful_owlv2_detection():
    test_image = "000000039769.jpg"
    prompts = ["a photo of a cat", "a photo of a dog"]

    image = Image.open(f"tests/tools/data/owlv2/{test_image}")

    owlv2 = Owlv2()

    prompts = ["a photo of a cat", "a photo of a dog"]

    results = owlv2(image, prompts=prompts)

    assert len(results) > 0

    for pred in results:
        assert pred.label == "a photo of a cat"
