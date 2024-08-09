import numpy as np
from PIL import Image
import torch

from vision_agent_tools.tools.text_video_classifier import TextVideoClassifier



def test_successful_clip_similarity_target_text_siglip():
    """
    This test verifies that TextVideoClassifier returns a valid iresponse when passed a target_text
    """
    image = np.array(Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB"))
    zeros = np.zeros(image.shape, dtype=np.uint8)

    test_video = np.array([zeros, image, zeros], dtype=np.uint8)

    siglip_class = TextVideoClassifier(device="cuda" if torch.cuda.is_available() else "cpu", model="siglip")

    results = siglip_class(video=test_video, target_text=["not tomatoes", "tomatoes"])

    assert len(results) == 3
    assert results[0][0] > results[0][1]
    assert results[1][0] < results[1][1]


def test_successful_clip_similarity_target_text_clip():
    """
    This test verifies that TextVideoClassifier returns a valid iresponse when passed a target_text
    """
    image = np.array(Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB"))
    zeros = np.zeros(image.shape, dtype=np.uint8)

    test_video = np.array([zeros, image, zeros], dtype=np.uint8)

    clip_class = TextVideoClassifier(device="cuda" if torch.cuda.is_available() else "cpu", model="clip")

    results = clip_class(video=test_video, target_text=["not tomatoes", "tomatoes"])

    assert len(results) == 3
