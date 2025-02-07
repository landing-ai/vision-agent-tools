import numpy as np
from PIL import Image
from vision_agent_tools.tools.image_video_similarity import ImageVideoSimilarity


def test_image_video_sim_siglip():
    image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")
    np_image = np.array(image)
    zeros = np.zeros(np_image.shape, dtype=np.uint8)

    test_video = np.array([zeros, np_image, zeros], dtype=np.uint8)
    siglip_sim = ImageVideoSimilarity(model="siglip")

    results = siglip_sim(video=test_video, target_image=image)
    assert len(results) == 3
    assert results[1] > results[0]

def test_image_video_sim_clip():
    image = Image.open("tests/tools/data/loca/tomatoes.jpg").convert("RGB")
    np_image = np.array(image)
    zeros = np.zeros(np_image.shape, dtype=np.uint8)

    test_video = np.array([zeros, np_image, zeros], dtype=np.uint8)
    clip_sim = ImageVideoSimilarity(model="clip")

    results = clip_sim(video=test_video, target_image=image)
    assert len(results) == 3
    assert results[1] > results[0]
