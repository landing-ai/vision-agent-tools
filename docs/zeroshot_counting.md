# LOCA (Low-shot Object Counting network with iterative prototype Adaptation).

This example demonstrates using the zeroshot LOCA tool for object counting in images.



```python
from vision_agent_tools.tools.zeroshot_counting import ZeroShotCounting

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# Load the image
image = Image.open(test_image)
# Initialize the counting model and choose the image output size you expect.
ObjectCounting = ZeroShotCounting(img_size=512)

# Run the inference
results = ObjectCounting(image, bbox=[12, 34, 56, 78])

# Let's find out how many objects were found in total
print("Found a total count of {results.count} objects on the image!")
```

::: vision_agent_tools.tools.zeroshot_counting
