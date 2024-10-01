# Flux1 

This example demonstrates using the Flux1 model to perform tasks such as image generation and mask inpainting based on text prompts. 

```python
import torch
from PIL import Image
from vision_agent_tools.models.flux1 import Flux1, Flux1Config, Flux1Task

# To perform image generation

# Set the prompt for image generation
prompt = "A purple car in a futuristic cityscape"

# Configure the Flux1 model
flux_config = Flux1Config(
    height=512,  # Image dimensions
    width=512,
    num_inference_steps=10,  # Number of steps for inference, higher means more detail
)

# Initialize the Flux1 model
flux_model = Flux1(model_config=flux_config)

# Generate an image from the prompt
generated_image = flux_model(
    task=Flux1Task.IMAGE_GENERATION,  # Image Generation Task
    prompt=prompt,
)

# Save the generated image
generated_image.save("generated_car.png")

#--------------------------------------------------------------------

# Alternatively, perform mask inpainting

# Suppose you have a cat image named "cat_image.jpg" that you want to use for mask inpainting
image_to_edit = Image.open("path/to/your/cat_image.jpg").convert("RGB")  # Image to inpaint

# Make sure to provide another image with the mask delimitating the cat
mask_image = Image.open("path/to/your/mask.png")  # Mask image indicating areas to change

# Set a new prompt for inpainting
inpainting_prompt = "A cute dog"

# Inpaint the image using the mask
inpainted_image = flux_model(
    task=Flux1Task.MASK_INPAINTING,  # Image Mask Inpainting Task
    prompt=inpainting_prompt,
    image=image_to_edit,
    mask_image=mask_image,
)

inpainted_image.save("inpainted_dog_over_cat.png")

```

::: vision_agent_tools.models.flux1
