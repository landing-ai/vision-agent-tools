# Flux1 

This example demonstrates using the Flux1 model to perform tasks such as image generation and mask inpainting based on text prompts.

### Parameters

- task: The task to perform using the model - either image generation ("generation") or mask inpainting ("inpainting").
- prompt: The text prompt describing the desired modifications.
- config: The `Flux1Config` class allows you to configure the parameters for the Flux1 model.
- image (Image.Image): The original image to be modified - 
used for the mask inpainting and image to image tasks.
- mask_image (Image.Image): The mask image indicating areas to be inpainted - used for the mask inpainting task

#### Flux1Config

Below is an example of how to create and use a `Flux1Config` object:

```python
from vision_agent_tools.models.flux1 import Flux1Config

config = Flux1Config(
    height=512,
    width=512,
    num_inference_steps=28,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    max_sequence_length=512,
    seed=42
)
```

- height: The height in pixels of the generated image. Defaults to 512.
- width: The width in pixels of the generated image. Defaults to 512.
- num_inference_steps: The number of inference steps to perform. Defaults to 28.
- guidance_scale: Guidance scale as defined in Classifier-Free Diffusion Guidance. Defaults to 3.5.
- num_images_per_prompt: The number of images to generate per prompt. Defaults to 1.
- max_sequence_length: Maximum sequence length to use with the prompt. Defaults to 512.
- seed: Seed for the random number generator. If not provided, a random seed is used.
- strength: Indicates extent to transform the reference `image`.
Must be between 0 and 1. A value of 1 essentially ignores `image`.

## Perform image generation

```python
import torch
from PIL import Image
from vision_agent_tools.models.flux1 import Flux1, Flux1Task

# To perform image generation
flux1 = Flux1()

generated_image = flux_model(
    task=Flux1Task.IMAGE_GENERATION,  # Image Generation Task
    prompt="A purple car in a futuristic cityscape",
    config=config
)
generated_image.save("generated_car.png")
```

--------------------------------------------------------------------

## Perform mask inpainting

To perform mask inpainting, both the original image and the mask image need to be provided. These images have the same dimensions. The mask should clearly delineate the areas that you want to modify in the original image. Additionally, the inpainting process includes a strength  parameter, which controls the intensity of the modifications applied to the masked areas.

```python
import torch
from PIL import Image
from vision_agent_tools.models.flux1 import Flux1, Flux1Task

# You have a cat image named "cat_image.jpg" that you want to use for mask inpainting
image_to_edit = Image.open("path/to/your/cat_image.jpg").convert("RGB")  # Image to inpaint

# Make sure to provide a mask image with the same dimensions, delineating the cat
mask_image = Image.open("path/to/your/mask.png")  # Mask image indicating areas to change

# Set a new prompt for inpainting
inpainting_prompt = "A cute dog"

# To perform image mask inpainting
flux1 = Flux1()

inpainted_image = flux_model(
    task=Flux1Task.MASK_INPAINTING,  # Image Mask Inpainting Task
    prompt=inpainting_prompt,
    image=image_to_edit,
    mask_image=mask_image,
    config=config
)

inpainted_image.save("inpainted_dog_over_cat.png")

```

--------------------------------------------------------------------

## Perform image-to-image generation

To perform image-to-image generation, you need to provide an original image along with a text prompt describing the desired modifications. The original image serves as the base, and the model will generate a new image based on the prompt.

```python
import torch
from PIL import Image
from vision_agent_tools.models.flux1 import Flux1, Flux1Task

# You have an original image named "original_image.jpg" that you want to use for image-to-image generation
original_image = Image.open("path/to/your/original_image.jpg").convert("RGB")  # Original image

# Set a new prompt for image-to-image generation
image_to_image_prompt = "A sunny beach with palm trees"

# To perform image-to-image generation
flux1 = Flux1()

generated_image = flux_model(
    task=Flux1Task.IMAGE_TO_IMAGE,  # Image-to-Image Generation Task
    prompt=image_to_image_prompt,
    image=original_image,
    config=config
)

generated_image.save("generated_beach.png")
```

::: vision_agent_tools.models.flux1
