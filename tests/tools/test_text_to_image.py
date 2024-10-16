import pytest
from PIL import Image

from vision_agent_tools.models.flux1 import Flux1Task


@pytest.mark.skip(
    reason="FIX THIS TEST. This test is incorrect, the configs are not being used."
)
def test_successful_image_generation(tool):
    prompt = "cat wizard, Pixar style"

    result = tool(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
    )

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (32, 32)


@pytest.mark.skip(
    reason="FIX THIS TEST. This test is incorrect, the configs are not being used."
)
def test_successful_image_mask_inpainting(tool):
    prompt = "cat wizard, Pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")
    mask_image = Image.open("tests/shared_data/images/chihuahua_mask.png")

    result = tool(
        task=Flux1Task.MASK_INPAINTING,
        prompt=prompt,
        image=image,
        mask_image=mask_image,
    )

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (32, 32)


# @pytest.fixture(scope="session")
# def tool():
#     flux_config = Flux1Config(
#         height=32,
#         width=32,
#         num_inference_steps=1,
#         max_sequence_length=256,
#     )

#     return TextToImage(model=TextToImageModel.FLUX1, model_config=flux_config)
