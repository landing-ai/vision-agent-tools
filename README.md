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

### Easy way
```bash
make install
```
### Advanced usage

You can install by running `poetry install --extras "all"` to install all tools, or with
`poetry install --extras "owlv2 florence2"` to install specific tools such as `owlv2`
and `florence2`.

## Usage

### Models
Models in this project are machine learning models that perform specific tasks (like object detection and instance segmentation).

Here's a simple example of how to use the `Owlv2` model to detect objects in an image:
```python
from PIL import Image
from vision_agent_tools.models.owlv2 import Owlv2

# load image
image = Image.open("/path/to/my/image.png")
model = Owlv2()

detections = model(image=image, prompts=["cat"])
```

### Tools
Tools are higher-level abstractions that wrap around one or more models to accomplish specific tasks. Each tool is designed to work with different models via a dynamic model registry, allowing users to switch between models.

Here's an example of how to use the `TextToObjectDetection` tool to detect objects in an image based on text prompts:


```python
from PIL import Image
from vision_agent_tools.tools.text_to_object_detection import TextToObjectDetection

# load image
img_path = "/path/to/my/image.jpg"
image = Image.open(img_path)

# Initialize the text-to-object detection tool with the desired model
detector = TextToObjectDetection(model="owlv2")

# Run the detector with the image and a text prompt
detections = detector(image=image, prompts=["find dogs in the picture"])
```

In this example:

- `TextToObjectDetection` tool is initialized with the "florence2" model.
- The tool detects objects based on the text prompt "find dogs in the picture" and returns a list of `TextToObjectDetectionOutput` containing the detection results.


# Contributing

## Clone the repo and install it

```bash
poetry install
poetry run pre-commit install
```

## Adding new model code

Tools can be added in `vision_agent_tools/models`. Simply create a new python file with
the model name and add a class with the same name. The class should inherit from
`BaseMLModel` and implement the `__call__` method. Here's a simplified example for adding
Owlv2 from the transformers library:

```python
from Image import Image
from vision_agent_tools.shared_types import BaseMLModel
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

## Registering a model in the model registry

To use a model with your tool, you need to register it in the `model_registry`. This allows tools to dynamically load the correct model based on the model name provided at runtime. In the `model_registry.py` file: add the model to the `MODEL_REGISTRY` dictionary, mapping the string identifier to the model class. To avoid dependency issues caused by importing all models at once, use the `lazy_import` function.

```python
MODEL_REGISTRY: Dict[str, Callable[[], BaseMLModel]] = {
    "florence2": lambda: lazy_import(f"{MODELS_PATH}.florence2", "Florence2")(),
    "owlv2": lambda: lazy_import(f"{MODELS_PATH}.owlv2", "Owlv2")(), # Register the new Owlv2 model here
}
```

## Adding new tool code

You can easily add new tools to the vision_agent_tools/tools directory. Tools are designed to wrap around one or more machine learning models and perform specific tasks. Steps to add a new Tool:

1. **Create a Python File**: In the `vision_agent_tools/tools` directory, create a new Python file named after the tool you want to add (e.g., text_to_object_detection.py).
2. **Map the Models to Tool**: Associate the list of models that can perform some task creating an Enum inside your tool file:
    ```python
    class TextToObjectDetectionModel(str, Enum):
        FLORENCE2 = "florence2"
        OWLV2 = "owlv2"
    ```
3. **Implement the Tool Class**: Inside the new Python file, create a class with the same name as the file. This class should inherit from BaseTool and implement the `__call__` method.

```python
from typing import List, Any
from enum import Enum
from PIL import Image
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class

class TextToObjectDetectionModel(str, Enum):
    OWLV2 = "owlv2"  # Register the Owlv2 model here

class TextToObjectDetection(BaseTool):
    def __init__(self, model: TextToObjectDetectionModel):
        if model not in TextToObjectDetectionModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )
        model_class = get_model_class(model_name=model)
        model_instance = model_class()
        super().__init__(model=model_instance)

    def __call__(
        self, image: Image.Image, prompts: List[str], **model_config: Dict[str, Any]
    ) -> List[TextToObjectDetectionOutput]:
        result = self.model(image=image, prompts=prompts, **model_config)
        return result


```

This setup ensures that your tools can automatically select and use the correct model for any given task and avoid tools using models that do not match with their designated task.

## Adding new dependencies
After that you can add the dependencies as optional like so:

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

## Unit tests
Example of how to run a single unit test:
```bash
poetry run pytest -vvvv tests/tools/test_shared_model_manager.py::test_swap_model_in_gpu
```
