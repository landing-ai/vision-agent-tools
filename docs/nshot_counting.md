# LOCA (Low-shot Object Counting network with iterative prototype Adaptation).

This example demonstrates how to use the NShot LOCA tool for object counting in images.


```python
from vision_agent_tools.models.nshot_counting import NShotCounting

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# Load the image
image = Image.open(test_image)
# Initialize the counting model and choose the image output size you expect.
ObjectCounting = NShotCounting(zero_shot=False, img_size=512)

# Run the inference
results = ObjectCounting(image, bbox=[12, 34, 56, 78])

# Let's find out how many objects were found in total
print("Found a total count of {results.count} objects on the image!")
```

::: vision_agent_tools.models.nshot_counting
