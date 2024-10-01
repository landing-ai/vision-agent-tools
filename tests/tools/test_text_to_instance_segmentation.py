import numpy as np
import pytest
from PIL import Image
from vision_agent_tools.tools.text_to_instance_segmentation import (
    TextToInstanceSegmentationTool,
)


def test_successful_image_detection_segmentation():
    """
    This test verifies that TextToInstanceSegmentationTool returns a valid response when passed an image.
    """
    test_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")

    tool = TextToInstanceSegmentationTool()

    results = tool(prompts=["tomato"], image=test_image)

    # The dictionary should have only one key: 0
    assert len(results) == 1
    # The dictionary should have 23 instances of the tomato class
    assert len(results[0]) == 23

    for instance in results[0].values():
        assert len(instance.bbox) == 4
        assert np.all(
            [0 <= coord <= np.max(test_image.size[:2]) for coord in instance.bbox]
        )
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == test_image.size[::-1]
        assert instance.label == "tomato"


def test_successful_video_detection_segmentation():
    """
    This test verifies that TextToInstanceSegmentationTool returns a valid response when passed a video.
    """
    tomatoes_image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")
    img_size = tomatoes_image.size
    np_test_img = np.array(tomatoes_image, dtype=np.uint8)
    zeros = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    test_video = np.array([np_test_img, zeros])

    tool = TextToInstanceSegmentationTool()

    results = tool(prompts=["tomato"], video=test_video)

    # The disctionary should have 2 keys for the two frames in the video
    assert len(results) == 2
    # The first frame should have 23 instances of the tomato class
    assert len(results[0]) == 23
    # The second frame should not have any tomato class since it is all zeros
    assert len(results[1]) == 0
    # First frame
    for instance in results[0].values():
        assert isinstance(instance.mask, np.ndarray)
        assert instance.mask.shape == np_test_img.shape[:2]
        assert instance.label == "tomato"


def test_invalid_media_detection_segmentation():
    """
    This test verifies that TextToInstanceSegmentationTool raises an error if the media is not valid.
    """
    tool = TextToInstanceSegmentationTool()

    with pytest.raises(ValueError):
        tool(prompts=["tomato"], image="invalid media")

    with pytest.raises(ValueError):
        tool(prompts=["tomato"], video="invalid media")

    with pytest.raises(AssertionError):
        tool(prompts=["tomato"], video=np.array([1, 2, 3]))
