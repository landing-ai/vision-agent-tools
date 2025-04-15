# Gemini Image Editor 

This example demonstrates using the Gemini Image Editing model to perform tasks such as image generation and inpainting based on text prompts.

### Parameters

- prompt: The text prompt describing the desired modifications.
- image (Image.Image): The original image to be modified.


## Perform image generation

```python
from vision_agent_tools.models.gemini_image_editor import GeminiImageEditor

gemini_model = GeminiImageEditor()

generated_image = gemini_model(
    prompt="A purple car in a futuristic landscape",
)
Image.fromarray(generated_image).save("car.png")
```

--------------------------------------------------------------------

## Perform inpainting

To perform inpainting, an image must be provided.

```python
from vision_agent_tools.models.gemini_image_editor import GeminiImageEditor
from PIL import Image

image = Image.open("car.png")
gemini_model = GeminiImageEditor()

generated_image = gemini_model(
    prompt="Make this car purple and put it in a futuristic landscape",
    image=image,
)
Image.fromarray(generated_image).save("new_car.png")

```

::: vision_agent_tools.models.gemini-image-editor
