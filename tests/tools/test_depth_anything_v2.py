from PIL import Image

from vision_agent_tools.tools.depth_anything_v2 import DepthAnythingV2


def test_successful_depth_estimation():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/tools/data/loca/{test_image}")

    depth = DepthAnythingV2()
    results = depth(image)
    assert len(results.map) > 0
