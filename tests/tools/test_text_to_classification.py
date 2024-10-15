from PIL import Image
from vision_agent_tools.tools.text_to_classification import TextToClassification


def test_successful_text_to_classification_tool():
    test_image = "safework.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    text_to_class = TextToClassification(model="nsfw_classification")
    results = text_to_class(image)

    assert results.label == "normal"
