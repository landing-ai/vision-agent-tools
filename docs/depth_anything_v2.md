# Depth-Anything-V2 

This example demonstrates using the Depth-Anything-V2 tool for depth estimation on images.



```python
from vision_agent_tools.models.depth_anything_v2 import DepthAnythingV2

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# Load the image
image = Image.open(test_image)
# Initialize the depth map estimation model.
depth_estimate = DepthAnythingV2()

# Run the inference
results = depth_estimate(image)

# Let's print the obtained depth map
print(results.map)
```

::: vision_agent_tools.models.depth_anything_v2
