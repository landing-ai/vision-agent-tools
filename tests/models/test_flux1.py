import os

import pytest
from PIL import Image
from pydantic import ValidationError

from vision_agent_tools.models.flux1 import Flux1, Flux1Task, Flux1Config


def test_image_mask_inpainting(model):
    prompt = "cat wizard, Pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")
    mask_image = Image.open("tests/shared_data/images/chihuahua_mask.png")

    config = Flux1Config(
        height=32,
        width=32,
        num_inference_steps=1,
        seed=42,
    )

    result = model(
        task=Flux1Task.MASK_INPAINTING,
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        config=config,
    )

    assert result is not None
    assert len(result) == 1
    image = result[0]
    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_image_generation(model):
    prompt = "cat wizard, Pixar style"

    config = Flux1Config(
        height=32,
        width=32,
        num_inference_steps=1,
        seed=42,
    )

    result = model(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        config=config,
    )

    assert result is not None
    assert len(result) == 1
    image = result[0]
    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_fail_image_generation_dimensions(model):
    prompt = "cat wizard, Pixar style"

    try:
        config = Flux1Config(
            height=31,
            width=31,
            num_inference_steps=1,
            seed=42,
        )

        model(
            task=Flux1Task.IMAGE_GENERATION,
            prompt=prompt,
            config=config,
        )
    except ValidationError as e:
        assert (
            repr(e.errors()[0]["msg"])
            == "'Assertion failed, height and width must be multiples of 8.'"
        )
        assert repr(e.errors()[0]["type"]) == "'assertion_error'"
        assert (
            repr(e.errors()[1]["msg"])
            == "'Assertion failed, height and width must be multiples of 8.'"
        )
        assert repr(e.errors()[1]["type"]) == "'assertion_error'"


def test_fail_image_mask_size(model):
    prompt = "cat wizard, Pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")
    mask_image = Image.open("tests/shared_data/images/chihuahua_mask.png")
    mask_image = mask_image.resize((64, 64))

    config = Flux1Config(
        height=32,
        width=32,
        num_inference_steps=1,
        seed=42,
    )

    try:
        model(
            task=Flux1Task.MASK_INPAINTING,
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            config=config,
        )
    except ValueError as e:
        assert str(e) == "The image and mask image should have the same size."


def test_different_images_different_seeds(model):
    prompt = "cat wizard, Pixar style"

    result_1 = model(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        config=Flux1Config(
            height=32,
            width=32,
            num_inference_steps=1,
            seed=42,
        ),
    )

    result_2 = model(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        config=Flux1Config(
            height=32,
            width=32,
            num_inference_steps=1,
            seed=0,
        ),
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

    config = Flux1Config(
        height=32,
        width=32,
        num_inference_steps=1,
        num_images_per_prompt=3,
        seed=42,
    )

    result = model(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        config=config,
    )

    assert result is not None
    assert len(result) == 3
    for image in result:
        assert image.mode == "RGB"
        assert image.size == (32, 32)


def test_image_to_image(model):
    prompt = "pixar style"
    image = Image.open("tests/shared_data/images/chihuahua.png")

    config = Flux1Config(
        height=32,
        width=32,
        num_inference_steps=1,
        seed=42,
    )

    result = model(
        task=Flux1Task.IMAGE_TO_IMAGE,
        prompt=prompt,
        image=image,
        config=config,
    )

    assert result is not None
    assert len(result) == 1
    image = result[0]
    assert image.mode == "RGB"
    assert image.size == (32, 32)


@pytest.fixture(scope="module")
def model():
    return Flux1(hf_access_token=os.environ["HF_ACCESS_TOKEN"])
