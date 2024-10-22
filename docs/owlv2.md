# OWLv2 Open-World Localization

This example demonstrates using the Owlv2 tool for object detection in images based on text prompts.



```python
from vision_agent_tools.models.owlv2 import Owlv2

# (replace this path with your own!)
test_image = "path/to/your/image.jpg"

# What are you looking for? Write your detective prompts here!
prompts = ["a photo of a cat", "a photo of a dog"]

# Load the image and create your Owlv2 detective tool
image = Image.open(test_image)
owlv2 = Owlv2()

# Time to put Owlv2 to work! Let's see what it finds...
results = owlv2(image, prompts=prompts)[0]

# Did Owlv2 sniff out any objects? Let's see the results!
if results:
    for detection in results:
        print(f"Found it! It looks like a {detection['label']} with a confidence of {detection['score']:.2f}.")
        print(f"Here's where it's hiding: {detection['bbox']}")
else:
    print("Hmm, Owlv2 couldn't find anything this time. Maybe try a different prompt?")
```

::: vision_agent_tools.models.owlv2
