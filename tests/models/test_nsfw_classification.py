from PIL import Image
from vision_agent_tools.models.nsfw_classification import NSFWClassification


def test_successful_nsfw_classification():
    test_image = "safework.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    nsfw_classifier = NSFWClassification()
    results = nsfw_classifier(image)

    assert results.label == "normal"
