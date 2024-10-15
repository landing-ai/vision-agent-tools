import numpy as np
import pytest
from PIL import Image

from vision_agent_tools.models.florence2_sam2 import Florence2SAM2


def test_successful_florence2_sam2_image():
    """
    This test verifies that Florence2SAM2 returns a valid iresponse when passed an image
    """
    test_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")

    florence2_sam2 = Florence2SAM2()

    results = florence2_sam2(prompts=["tomato"], image=test_image)

    # The list should have only one value at position 0
    assert len(results) == 1
    # The list should have 22 instances of the tomato class
    assert len(results[0]) == 22
    for instance in results[0]:
        assert len(instance.bbox) == 4
        assert np.all(
            [0 <= coord <= np.max(test_image.size[:2]) for coord in instance.bbox]
        )
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == test_image.size[::-1]
        assert instance.label == "tomato"


def test_successful_florence2_sam2_video():
    """
    This test verifies that Florence2SAM2 returns a valid iresponse when passed a video
    """
    tomatoes_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    test_video = np.array([np_test_img, zeros])

    florence2_sam2 = Florence2SAM2()

    results = florence2_sam2(prompts=["tomato"], video=test_video)

    # The list should have 2 keys for the two frames in the video
    assert len(results) == 2
    # The first frame should have 22 instances of the tomato class
    assert len(results[0]) == 22
    # The second frame should not have any tomato class since it is all zeros
    assert len(results[1]) == 0
    # First frame
    for instance in results[0]:
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == np_test_img.shape[:2]
        assert instance.label == "tomato"


def test_successful_nms_florence2_sam2_video():
    """
    This test verifies that Florence2SAM2 returns a valid iresponse when passed a video
    with a nms_value
    """
    tomatoes_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    test_video = np.array([np_test_img, zeros])

    florence2_sam2 = Florence2SAM2()

    results = florence2_sam2(prompts=["tomato"], video=test_video, nms_threshold=0.1)

    # The list should have 2 keys for the two frames in the video
    assert len(results) == 2
    # The first frame should have 22 instances of the tomato class
    assert len(results[0]) == 22
    # The second frame should not have any tomato class since it is all zeros
    assert len(results[1]) == 0
    # First frame
    for instance in results[0]:
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == np_test_img.shape[:2]
        assert instance.label == "tomato"


def test_florence2_sam2_invalid_media():
    """
    This test verifies that Florence2SAM2 raises an error if the media is not a valid.
    """
    florence2_sam2 = Florence2SAM2()

    with pytest.raises(ValueError):
        florence2_sam2(prompts=["tomatoe"], image="invalid media")

    with pytest.raises(ValueError):
        florence2_sam2(prompts=["tomato"], video="invalid media")

    with pytest.raises(AssertionError):
        florence2_sam2(prompts=["tomatoe"], video=np.array([1, 2, 3]))
