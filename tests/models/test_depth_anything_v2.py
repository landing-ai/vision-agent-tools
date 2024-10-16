import numpy as np
from PIL import Image

from vision_agent_tools.models.depth_anything_v2 import DepthAnythingV2


def test_successful_depth_estimation():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    depth = DepthAnythingV2()
    results = depth(image)
    assert len(results.map) > 0


def test_depth_grayscale_true():
    test_image = "tomatoes.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    depth = DepthAnythingV2()

    results = depth(image=image, grayscale=True)

    # Check that results.map is a 3D numpy array
    assert isinstance(results.map, np.ndarray)
    assert results.map.ndim == 2, f"Expected 2D array, got {results.map.ndim}D array"

    # Check that image values are in the range [0, 255]
    assert np.all((results.map >= 0) & (results.map <= 255))
