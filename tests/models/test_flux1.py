from PIL import Image
from vision_agent_tools.models.flux1 import Flux1, Flux1Task


def test_successful_image_generation():
    prompt = "cat wizard, Pixar style, 8k"

    flux1 = Flux1()
    result = flux1(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=10,
        seed=42,
    )

    result.save("tests/models/data/flux1/cat_wizard.png")

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_successful_image_mask_inpainting():
    prompt = "cat wizard, Pixar style, 8k"
    image = Image.open("tests/models/data/flux1/chihuahua.png").convert("RGB")
    mask_image = Image.open("tests/models/data/flux1/chihuahua_mask.png")

    flux1 = Flux1()
    result = flux1(
        task=Flux1Task.MASK_INPAINTING,
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        height=512,
        width=512,
        num_inference_steps=10,
        guidance_scale=7,
        strength=0.85,
        seed=42,
    )

    result.save("tests/models/data/flux1/chihuahua_to_cat_wizard.png")

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_parameters_image_generation():
    prompt = "cat wizard, Pixar style, 8k"

    flux1 = Flux1()
    result = flux1(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        height=128,
        width=128,
        guidance_scale=0.5,
        num_inference_steps=10,
        seed=42,
    )

    result.save("tests/models/data/flux1/cat_1_inference_step.png")

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (128, 128)


def test_fail_image_generation_dimensions():
    prompt = "cat wizard, Pixar style, 8k"

    flux1 = Flux1()
    height = 500
    width = 500
    try:
        flux1(
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


def test_fail_image_mask_size():
    prompt = "cat wizard, Pixar style, 8k"
    image = Image.open("tests/models/data/flux1/chihuahua.png").convert("RGB")
    mask_image = Image.open("tests/models/data/flux1/chihuahua_mask.png")
    mask_image = mask_image.resize((128, 128))

    flux1 = Flux1()
    try:
        flux1(
            task=Flux1Task.MASK_INPAINTING,
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=512,
            width=128,
            num_inference_steps=1,
            seed=42,
        )
    except ValueError as e:
        assert str(e) == "The image and mask image should have the same size."


def test_different_images_different_seeds():
    prompt = "cat wizard, Pixar style, 8k"

    flux1 = Flux1()
    result_1 = flux1(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=1,
        seed=42,
    )

    result_2 = flux1(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=1,
        seed=0,
    )

    assert result_1 is not None
    assert result_2 is not None
    assert result_1 != result_2


def test_multiple_images_per_prompt():
    prompt = "cat wizard, Pixar style, 8k"

    flux1 = Flux1()
    result = flux1(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
        height=128,
        width=128,
        num_inference_steps=1,
        num_images_per_prompt=3,
        seed=42,
    )

    assert result is not None
    assert len(result) == 3
    for image in result:
        assert image.mode == "RGB"
        assert image.size == (128, 128)
