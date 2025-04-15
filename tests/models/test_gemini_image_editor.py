import pytest
from PIL import Image

from vision_agent_tools.models.gemini_image_editor import GeminiImageEditor


def test_image_inpainting(model):
    prompt = "cat wizard, Pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")

    result = model(
        prompt=prompt,
        image=image,
    )

    assert result is not None


@pytest.fixture(scope="module")
def model():
    return GeminiImageEditor()
