from PIL import Image
from vision_agent_tools.models.flux1 import Flux1, Flux1Config, Flux1Task


def test_successful_image_generation():
    prompt = "cat wizard, Pixar style, 8k"

    flux_config = Flux1Config(
        height=512,
        width=512,
        num_inference_steps=1,
        max_sequence_length=256,
    )

    flux1 = Flux1(model_config=flux_config)

    result = flux1(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
    )

    result.save("tests/models/data/flux1/cat_wizard.png")

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_successful_image_mask_inpainting():
    prompt = "cat wizard, Pixar style, 8k"
    image = Image.open("tests/models/data/flux1/chihuahua.png").convert("RGB")
    mask_image = Image.open("tests/models/data/flux1/chihuahua_mask.png")

    flux_config = Flux1Config(
        height=512,
        width=512,
        num_inference_steps=1,
        max_sequence_length=256,
    )

    flux1 = Flux1(model_config=flux_config)

    result = flux1(
        task=Flux1Task.MASK_INPAINTING,
        prompt=prompt,
        image=image,
        mask_image=mask_image,
    )

    result.save("tests/models/data/flux1/chihuahua_to_cat_wizard.png")

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (512, 512)
