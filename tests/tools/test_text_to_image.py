from PIL import Image

from vision_agent_tools.models.flux1 import Flux1Task


def test_successful_image_mask_inpainting(tool):
    prompt = "cat wizard, Pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")
    mask_image = Image.open("tests/shared_data/images/chihuahua_mask.png")

    result = tool(
        task=Flux1Task.MASK_INPAINTING,
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=7,
        strength=0.85,
        seed=42,
    )

    assert result is not None
    assert len(result) == 1
    image = result[0]
    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_successful_image_generation(tool):
    prompt = "cat wizard, Pixar style"

    result = tool(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        height=32,
        width=32,
        guidance_scale=0.5,
        num_inference_steps=1,
        seed=42,
    )

    assert result is not None
    assert len(result) == 1
    image = result[0]
    assert image.mode == "RGB"
    assert image.size == (32, 32)
