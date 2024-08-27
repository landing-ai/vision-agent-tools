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

    # The disctionary should have only one key: 0
    assert len(results) == 1
    # The dictionary should have 23 instances of the tomato class
    assert len(results[0]) == 23
    for instance in results[0].values():
        assert len(instance.bounding_box) == 4
        assert np.all(
            [
                0 <= coord <= np.max(test_image.size[:2])
                for coord in instance.bounding_box
            ]
        )
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == test_image.size[::-1]
        assert instance.label == "tomato"


def test_successful_florence2_sam2_video():
    """
    This test verifies that Florence2SAM2 returns a valid iresponse when passed a video
    """
    tomatoes_image = np.array(
        Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB"), dtype=np.uint8
    )
    test_video = np.array(
        [tomatoes_image, np.zeros(tomatoes_image.shape, dtype=np.uint8)]
    )

    florence2_sam2 = Florence2SAM2()

    results = florence2_sam2(prompts=["tomato"], video=test_video)

    # The disctionary should have 2 keys for the two frames in the video
    assert len(results) == 2
    # The first frame should have 23 instances of the tomato class
    assert len(results[0]) == 23
    assert len(results[1]) == 23
    # First frame
    for instance in results[0].values():
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == tomatoes_image.shape[:2]
        assert instance.label == "tomato"

    # Second frame
    for instance in results[1].values():
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == tomatoes_image.shape[:2]
        assert instance.label == "tomato"
        # All masks should de empty since it's a black frame
        assert np.all(instance.mask == 0)


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
