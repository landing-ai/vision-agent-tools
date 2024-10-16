import pytest
from PIL import Image

from vision_agent_tools.models.flux1 import Flux1, Flux1Task


def test_image_mask_inpainting(model):
    prompt = "cat wizard, Pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")
    mask_image = Image.open("tests/shared_data/images/chihuahua_mask.png")

    result = model(
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


def test_image_generation(model):
    prompt = "cat wizard, Pixar style"

    result = model(
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


def test_fail_image_generation_dimensions(model):
    prompt = "cat wizard, Pixar style"

    height = 31
    width = 31
    try:
        model(
            task=Flux1Task.IMAGE_GENERATION,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=1,
            seed=42,
        )
    except ValueError as e:
        assert (
            str(e)
            == f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
        )


def test_fail_image_mask_size(model):
    prompt = "cat wizard, Pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")
    mask_image = Image.open("tests/shared_data/images/chihuahua_mask.png")
    mask_image = mask_image.resize((64, 64))

    try:
        model(
            task=Flux1Task.MASK_INPAINTING,
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=32,
            width=32,
            num_inference_steps=1,
            seed=42,
        )
    except ValueError as e:
        assert str(e) == "The image and mask image should have the same size."


def test_different_images_different_seeds(model):
    prompt = "cat wizard, Pixar style"

    result_1 = model(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        height=32,
        width=32,
        num_inference_steps=1,
        seed=42,
    )

    result_2 = model(
        prompt=prompt,
        height=32,
        width=32,
        num_inference_steps=1,
        seed=0,
    )

    assert result_1 is not None
    assert result_2 is not None
    assert len(result_1) == 1
    assert len(result_2) == 1
    image_1 = result_1[0]
    image_2 = result_2[0]
    assert image_1.mode == "RGB"
    assert image_1.size == (32, 32)
    assert image_2.mode == "RGB"
    assert image_2.size == (32, 32)
    assert image_1 != image_2


def test_multiple_images_per_prompt(model):
    prompt = "cat wizard, Pixar style"

    result = model(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        height=32,
        width=32,
        num_inference_steps=1,
        num_images_per_prompt=3,
        seed=42,
    )

    assert result is not None
    assert len(result) == 3
    for image in result:
        assert image.mode == "RGB"
        assert image.size == (32, 32)


@pytest.fixture(scope="session")
def model():
    return Flux1()
