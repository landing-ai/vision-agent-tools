# NSFW (Not Safe for Work) classification

This example demonstrates using the Not Safe for Work classification tool.


```python
from vision_agent_tools.models.nsfw_classification import NSFWClassification

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# Load the image
image = Image.open(test_image)
# Initialize the NSFW model.
nsfw_classification = NSFWClassification()

# Run the inference
results = nsfw_classification(image)

# Let's print the predicted label
print(results.label)
```

::: vision_agent_tools.models.nsfw_classification
