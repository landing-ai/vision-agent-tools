import pytest
from PIL import Image

from vision_agent_tools.models.internlm_xcomposer2 import InternLMXComposer2


def test_internlm_xcomposer2_video(shared_model, random_video_generator):
    video_np = random_video_generator(n_frames=10)
    prompt = "Here are some frames of a video. Describe this video in detail"

    answer = shared_model(video=video_np, prompt=prompt, chunk_length=6)

    assert len(answer) == 2


def test_internlm_xcomposer2_image(shared_model):
    test_image = "car.jpg"
    image = Image.open(f"tests/shared_data/images/{test_image}")

    answer = shared_model(image=image, prompt="what is the color of the car?")

    assert len(answer) > 0


@pytest.fixture(scope="module")
def shared_model():
    return InternLMXComposer2()
