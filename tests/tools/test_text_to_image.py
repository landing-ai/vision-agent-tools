from PIL import Image
from vision_agent_tools.tools.text_to_image import TextToImage, TextToImageModel
from vision_agent_tools.models.flux1 import Flux1Task, Flux1Config


def test_successful_image_generation():
    prompt = "cat wizard, Pixar style, 8k"

    flux_config = Flux1Config(
        height=512,
        width=512,
        num_inference_steps=1,
        max_sequence_length=256,
    )

    tool = TextToImage(model=TextToImageModel.FLUX1, model_config=flux_config)

    result = tool(
        task=Flux1Task.IMAGE_GENERATION,
        prompt=prompt,
    )

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (512, 512)


def test_successful_image_mask_inpainting():
    prompt = "cat wizard, Pixar style, 8k"
    image = Image.open("tests/shared_data/images/chihuahua.png").convert("RGB")
    mask_image = Image.open("tests/shared_data/images/chihuahua_mask.png")

    flux_config = Flux1Config(
        height=512,
        width=512,
        num_inference_steps=1,
        max_sequence_length=256,
    )

    tool = TextToImage(model=TextToImageModel.FLUX1, model_config=flux_config)

    result = tool(
        task=Flux1Task.MASK_INPAINTING,
        prompt=prompt,
        image=image,
        mask_image=mask_image,
    )

    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (512, 512)
