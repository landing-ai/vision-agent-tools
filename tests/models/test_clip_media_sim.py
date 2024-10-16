import numpy as np
from PIL import Image
import torch

from vision_agent_tools.models.clip_media_sim import CLIPMediaSim


def test_successful_clip_similarity_target_image(random_video_generator):
    """
    This test verifies that CLIPMediaSim returns a valid iresponse when passed a target_text
    """
    test_video = random_video_generator()
    test_target_image = Image.fromarray(test_video[1])

    clip_sim = CLIPMediaSim(device="cuda" if torch.cuda.is_available() else "cpu")

    results = clip_sim(video=test_video, target_image=test_target_image)

    # apparently generating random images generates similar embeddings for all frames
    # so the similarity is high for all frames.
    assert len(results) == 1
    assert all(len(result) == 2 for result in results)
    # Should match with frame in index 1
    assert results[0][0] == 1


def test_successful_clip_similarity_target_text():
    """
    This test verifies that CLIPMediaSim returns a valid iresponse when passed a target_text
    """
    image = np.array(Image.open("tests/shared_data/images/tomatoes.jpg").convert("RGB"))
    zeros = np.zeros(image.shape, dtype=np.uint8)

    test_video = np.array([zeros, image, zeros], dtype=np.uint8)

    clip_sim = CLIPMediaSim(device="cuda" if torch.cuda.is_available() else "cpu")

    results = clip_sim(video=test_video, target_text="tomatoes")

    assert len(results) == 1
    assert all(len(result) == 2 for result in results)
    # Should match with frame in ineex 1 that has the picture of tomatoes
    assert results[0][0] == 1


def test_only_one_target(random_video_generator):
    """
    This test verifies that CLIPMediaSim raises a ValueError if both target_image and target_text are provided.
    """
    test_video = random_video_generator()
    test_target_image = Image.fromarray(test_video[1])

    clip_sim = CLIPMediaSim(device="cuda" if torch.cuda.is_available() else "cpu")

    try:
        clip_sim(
            video=test_video, target_image=test_target_image, target_text="random image"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised")
