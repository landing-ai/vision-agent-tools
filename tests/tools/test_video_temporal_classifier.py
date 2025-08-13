import numpy as np
from PIL import Image
from vision_agent_tools.tools.text_video_temporal_classifier import TextVideoTemporalClassifier


def test_successful_text_video_tmp_classifier():
    image = np.array(Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB"))
    zeros = np.zeros(image.shape, dtype=np.uint8)

    test_video = np.array([zeros, image, zeros], dtype=np.uint8)
    text_class = TextVideoTemporalClassifier()
    results = text_class("tomatoe", test_video, chunk_size=1)
    assert results == [0, 1, 0]
