# Vision tools

This repository contains tools that solve vision problems. This tools can be used in conjunction with the [vision-agent](https://github.com/landing-ai/vision-agent).

# Use the tools

You can use single tools by instantiating them and calling the tool with a dictionary object.

```python
from PIL import Image
from vision_agent_tools.tools.qr_reader import QRReader

# load image
image = Image.open("/path/to/my/image.png")

qr_reader = QRReader()

qr_detections = qr_reader(image=image)
```

# Contributing

## Clone the repo and install it

```bash
>>> poetry install
>>> poetry run pre-commit install
```

## Add a new tool

Tools can be added in `vision_agent_tools/tools`. Simply create a new python file with
the tool name and add a class with the same name. The class should inherit from
`BaseTool` and implement the `__call__` method. For example

```python
from Image import Image
from vision_agent_tools.base_tool import BaseTool

class MyTool(BaseTool):
    def __init__(self):
        self.model = load_model()
    def __call__(self, image: Image.Image):
        output = model(image)
        return {"result": output}
```

You can add dependencies by adding them as optional:

```bash
poetry add <dependency> --optional
```

After adding each dependency, you need to go to the `pyproject.toml` file and add a new
group under `[tool.poetry.extras]`. This will allow the installation of the package with
specific tools, like `pip install "vision-agent-tools[qr-reader]"`. You also need to
manually add each dependency to the "all" group so that user can install all tools as
`pip install "vision-agent-tools[all]"`. Example for the "qr-reader" tool:

```toml
[tool.poetry.extras]
all = ["qreader"]
qr-reader = ["qreader"]
```

