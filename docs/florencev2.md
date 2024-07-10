# Florence-2

This example demonstrates using the Florence2 tool to interpret simple text prompts to perform tasks like captioning, object detection, and segmentation.

__NOTE__: The Florence-2 model can only be used in GPU environments.

```python
from vision_agent_tools.tools.florencev2 import Florencev2, PromptTask

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# Choose the task that you are planning to use
task_prompt = PromptTask.CAPTION

# Load the image and create initialize the Florencev2 model
image = Image.open(test_image)
run_florence = Florencev2()

# Time to put Florencev2 to work! Let's see what it finds...
results = run_florence(image, task=task_prompt)

# Print the output result
print(f"The image contains: {results[task_prompt]}")
```

::: vision_agent_tools.tools.florencev2
