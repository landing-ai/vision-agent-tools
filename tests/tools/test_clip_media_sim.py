import numpy as np
from PIL import Image
import torch

from vision_agent_tools.tools.clip_media_sim import CLIPMediaSim


def _generate_test_video():
    """
    Generate a test video with 3 frames.
    """
    return np.random.randint(0, 255, (3, 300, 300, 3), dtype=np.uint8)


def test_successful_clip_similarity_target_image():
    """
    This test verifies that CLIPMediaSim returns a valid iresponse when passed a target_text
    """
    test_video = _generate_test_video()
    test_target_image = Image.fromarray(test_video[1])

    clip_sim = CLIPMediaSim(device="cuda" if torch.cuda.is_available() else "cpu")

    results = clip_sim(
        video=test_video, timestamps=[0.0, 1.0, 2.0], target_image=test_target_image
    )

    # apparently generating random images generates similar embeddings for all frames
    # so the similarity is high for all frames.
    assert len(results) == 1
    assert all(len(result) == 2 for result in results)
    # Should match with frame in timestamp 1.0 (index 1)
    assert results[0][0] == 1.0


def test_successful_clip_similarity_target_text():
    """
    This test verifies that CLIPMediaSim returns a valid iresponse when passed a target_text
    """
    image = np.array(Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB"))
    zeros = np.zeros(image.shape, dtype=np.uint8)

    test_video = np.array([zeros, image, zeros], dtype=np.uint8)

    clip_sim = CLIPMediaSim(device="cuda" if torch.cuda.is_available() else "cpu")

    results = clip_sim(
        video=test_video, timestamps=[0.0, 1.0, 2.0], target_text="tomatoes"
    )

    assert len(results) == 1
    assert all(len(result) == 2 for result in results)
    # Should match with frame in timestamp 1.0 (index 1) that has the picture of tomatoes
    assert results[0][0] == 1.0


def test_only_one_target():
    """
    This test verifies that CLIPMediaSim raises a ValueError if both target_image and target_text are provided.
    """
    test_video = _generate_test_video()
    test_target_image = Image.fromarray(test_video[1])

    clip_sim = CLIPMediaSim(device="cuda" if torch.cuda.is_available() else "cpu")

    try:
        clip_sim(
            video=test_video,
            timestamps=[0.0, 1.0, 2.0],
            target_image=test_target_image,
            target_text="random image",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised")
