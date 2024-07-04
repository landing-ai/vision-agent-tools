from PIL import Image

from vision_agent_tools.tools.zeroshot_counting import ZeroShotCounting


def test_successful_zeroshot_counting():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/tools/data/loca/{test_image}")

    count = ZeroShotCounting()
    bbox = [267, 44, 324, 82]
    results = count(image, bbox=bbox)
    assert results.count > 0
    assert len(results.masks) > 0