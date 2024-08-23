<img alt="vision_agent" src="https://landing.ai/wp-content/uploads/2024/06/LightLogo.svg" />


<div align="center">
    <img alt="vision_agent" height="200px" src="https://github.com/landing-ai/vision-agent-tools/blob/main/assets/vat.png?raw=true">

# 🔍🤖 Vision Agent Tools
[![](https://dcbadge.vercel.app/api/server/wPdN8RCYew?compact=true&style=flat)](https://discord.gg/wPdN8RCYew)
![docs_status](https://github.com/landing-ai/vision-agent-tools/actions/workflows/publish_docs.yml/badge.svg)
![tests_status](https://github.com/landing-ai/vision-agent-tools/actions/workflows/unit-tests.yml/badge.svg)
</div>

## Unleash the Power of Computer Vision!

This repository provides a suite of tools designed to tackle your image and video-based computer vision challenges. Whether you're working on object detection, image classification, QR reading, counting items, or other visual tasks, these tools can streamline your development process.

### Key Features:

- Image & Video Support.
- Detailed Documentation: Get started quickly and explore advanced features with our documentation: https://landing-ai.github.io/vision-agent-tools/.
- Seamless Integration: These tools are designed to work in conjunction with the powerful [Vision Agent](https://github.com/landing-ai/vision-agent).

# Ready to Get Started?

For a quick and easy introduction to the core functionalities, head over to the Vision Agent web app: https://va.landing.ai/tool. This is a great starting point to get familiar with the capabilities and potential of the tools before diving deeper into the code.

Let's Build Something Amazing!

We encourage you to explore the tools, leverage the documentation, and contribute to the project.


## Installation
You can install by running `poetry install --extras "all"` to install all tools, or with
`poetry install --extras "owlv2 florencev2"` to install specific tools such as `owlv2`
and `florencev2`.

## Usage
Here's a simple example of how to use the `Owlv2` tool to detect objects in an image:
```python
from PIL import Image
from vision_agent_tools.models.owlv2 import Owlv2

# load image
image = Image.open("/path/to/my/image.png")
model = Owlv2()

detections = model(image=image, prompts=["cat"])
```

# Contributing

## Clone the repo and install it

```bash
poetry install
poetry run pre-commit install
```

## Adding new tool code

Tools can be added in `vision_agent_tools/tools`. Simply create a new python file with
the tool name and add a class with the same name. The class should inherit from
`BaseMLModel` and implement the `__call__` method. Here's a simplified example for adding
Owlv2 from the transformers library:

```python
from Image import Image
from vision_agent_tools.base_tool import BaseMLModel
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class Owlv2(BaseMLModel):
    def __init__(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    def __call__(self, image: Image.Image, prompt: list[str]):
        inputs = self.processor(image, [prompt], return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor(image.size[::-1])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)
        output = []
        for box, score, label in zip(resuts[0]["boxes"], results[0]["scores"], results[0]["labels"]):
            output.append({"box": box.tolist()), "score": score.item(), "label": label.item()}
        return output
```

## Adding new dependencies
Afer that you can add the dependencies as optional like so:

```bash
poetry add transformers --optional
```

After adding each dependency, you need to go to the `pyproject.toml` file and add a new
group under `[tool.poetry.extras]`. This will allow the installation of the package with
specific tools.
```toml
[tool.poetry.extras]
all = ["transformers"]
owlv2 = ["transformers"]
```

Here we've added `"transformers"` as the dependency for the `owlv2` group. With these
you can now install just tools you need by running:
```bash
poetry install -E "owlv2"
```

or installing everything with:
```bash
poetry install -E "all"
```
