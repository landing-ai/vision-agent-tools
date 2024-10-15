from PIL import Image

from vision_agent_tools.tools.depth_estimation import DepthEstimation


def test_successful_depth_estimation():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    tool = DepthEstimation(model="depth_anything_v2")
    output = tool(image=image)

    assert output.map.size > 0
