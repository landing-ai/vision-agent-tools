from PIL import Image

from vision_agent_tools.tools.depth_estimation import DepthEstimation


def test_successful_zeroshot_counting():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/tools/data/loca/{test_image}")

    count = DepthEstimation()
    results = count(image)
    assert results.map > 0
