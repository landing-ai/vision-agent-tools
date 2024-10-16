from PIL import Image

from vision_agent_tools.models.controlnet_aux import Image2Pose


def test_successful_image_2_pos_detection():
    """
    This test verifies that Image2Pose returns a valid image for a valid input image.
    """
    test_image = "pose.png"
    image_path = f"tests/shared_data/images/{test_image}"

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise AssertionError(f"Test image '{test_image}' not found!") from None

    image_2_pose = Image2Pose()

    # Run pose detection on the image
    results = image_2_pose(image)

    assert isinstance(results, Image.Image)
    assert results.mode == "RGB"
    assert results.size == (563, 855)
