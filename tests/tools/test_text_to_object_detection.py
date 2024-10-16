from PIL import Image

from vision_agent_tools.tools.text_to_object_detection import (
    TextToObjectDetection,
    TextToObjectDetectionModel,
)
from vision_agent_tools.models.owlv2 import OWLV2Config


def test_successful_text_to_object_detection_owlv2():
    test_image = "000000039769.jpg"
    prompts = ["a photo of a cat", "a photo of a dog"]

    image = Image.open(f"tests/shared_data/images/{test_image}")

    tool = TextToObjectDetection(model=TextToObjectDetectionModel.OWLV2)
    output = tool(image=image, prompts=prompts)[0]

    assert len(output) > 0

    for pred in output:
        assert pred.label == "a photo of a cat"


def test_successful_text_to_object_detection_florence2():
    test_image = "000000039769.jpg"
    prompts = ["cat", "dog"]

    image = Image.open(f"tests/shared_data/images/{test_image}")

    tool = TextToObjectDetection(model=TextToObjectDetectionModel.FLORENCE2)
    output = tool(image=image, prompts=prompts)[0]

    assert len(output) > 0

    for pred in output:
        assert pred.label in ["cat", "dog"]


def test_successful_text_to_object_detection_custom_confidence():
    test_image = "000000039769.jpg"
    prompts = ["a photo of a cat", "a photo of a dog"]

    image = Image.open(f"tests/shared_data/images/{test_image}")

    tool = TextToObjectDetection(
        model=TextToObjectDetectionModel.OWLV2, model_config=OWLV2Config(confidence=0.2)
    )
    output = tool(image=image, prompts=prompts)[0]

    assert len(output) > 0

    for pred in output:
        assert pred.label == "a photo of a cat"
