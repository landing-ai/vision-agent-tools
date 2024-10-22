# FlorenceQA

This example demonstrates using the Florence2-QA tool to   to answer questions about images.

__NOTE__: The FlorenceQA model can only be used in GPU environments.

```python
from vision_agent_tools.models.florence2_qa import FlorenceQA

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# Load the image and create initialize the FlorenceQA model
image = Image.open(test_image)
run_florence_qa = FlorenceQA()

# Time to put FlorenceQA to work! Let's pose a question about the image
answer = run_florence_qa(image, question="Is there a dog in the image?")

# Print the output answer
print(answer)
```

::: vision_agent_tools.models.florence2_qa
