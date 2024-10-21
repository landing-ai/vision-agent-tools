import pytest
from PIL import Image

from vision_agent_tools.tools.text_to_object_detection import (
    TextToObjectDetection,
    TextToObjectDetectionModel,
)
from vision_agent_tools.models.owlv2 import OWLV2Config


def test_text_to_object_detection_owlv2(shared_tool_owlv2):
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompt = "a photo of a cat, a photo of a dog"

    response = shared_tool_owlv2(prompt, images=[image])

    # assert len(output) > 0

    # for pred in output:
    #     assert pred.label == "a photo of a cat"


def test_text_to_object_detection_florence2(shared_tool_florence2):
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompt = "cat, dog"

    response = shared_tool_florence2(prompt, images=[image])

    # assert len(output) > 0

    # for pred in output:
    #     assert pred.label in ["cat", "dog"]


def test_successful_text_to_object_detection_custom_confidence():
    image_path = "tests/shared_data/images/000000039769.jpg"
    image = Image.open(image_path)
    prompt = "a photo of a cat, a photo of a dog"

    tool = TextToObjectDetection(
        model=TextToObjectDetectionModel.OWLV2, model_config=OWLV2Config(confidence=1.0)
    )
    response = tool(prompt, images=[image])

    # assert len(output) > 0

    # for pred in output:
    #     assert pred.label == "a photo of a cat"


@pytest.fixture(scope="session")
def shared_tool_owlv2():
    return TextToObjectDetection(model=TextToObjectDetectionModel.OWLV2)


@pytest.fixture(scope="session")
def shared_tool_florence2():
    return TextToObjectDetection(model=TextToObjectDetectionModel.FLORENCE2)
