from PIL import Image

from vision_agent_tools.tools.text_to_object_detection import TextToObjectDetection


def test_successful_text_to_object_detection():
    test_image = "000000039769.jpg"
    prompts = ["a photo of a cat", "a photo of a dog"]

    image = Image.open(f"tests/tools/data/owlv2/{test_image}")

    tool = TextToObjectDetection(model="owlv2")
    output = tool(image=image, prompts=prompts)[0].output

    assert len(output) > 0

    for pred in output[0]:
        assert pred.label == "a photo of a cat"
