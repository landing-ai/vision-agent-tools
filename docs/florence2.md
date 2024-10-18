# Florence-2

This example demonstrates using the Florence2 tool to interpret simple text prompts to perform tasks like captioning, object detection, and segmentation.

```python
from vision_agent_tools.shared_types import PromptTask
from vision_agent_tools.models.florence2 import Florence2

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# Choose the task that you are planning to use
task_prompt = PromptTask.CAPTION

# Load the image and create initialize the Florence2 model
image = Image.open(test_image)
model = Florence2()

# Time to put Florence2 to work! Let's see what it finds...
results = model(images=[image], task=task_prompt)

# Print the output result
print(f"The image contains: {results[0]}")
```

::: vision_agent_tools.models.florence2
