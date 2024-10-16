from PIL import Image

from vision_agent_tools.models.nshot_counting import NShotCounting


def test_successful_zeroshot_counting():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    count = NShotCounting()
    results = count(image)
    assert results.count > 0
    assert len(results.heat_map) > 0


def test_successful_nshot_counting():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    count = NShotCounting(zero_shot=False)
    bbox = [267, 44, 324, 82]
    results = count(image, bbox=bbox)
    assert results.count > 0
    assert len(results.heat_map) > 0
