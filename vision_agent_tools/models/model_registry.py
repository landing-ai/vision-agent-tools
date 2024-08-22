from typing import Dict, Callable
from vision_agent_tools.shared_types import BaseMLModel
from vision_agent_tools.models.florencev2 import Florencev2


MODEL_REGISTRY: Dict[str, Callable[[], BaseMLModel]] = {"florence2": Florencev2}


def get_model(model_name: str) -> BaseMLModel:
    """
    Retrieve a model from the registry based on the model name
    """

    model_class = MODEL_REGISTRY.get(model_name)

    if not model_class:
        raise ValueError(f"Model '{model_name}' is not registered")

    return model_class()
