from typing import Dict, Callable
from vision_agent_tools.shared_types import BaseMLModel
from vision_agent_tools.models.florencev2 import Florencev2
from vision_agent_tools.models.owlv2 import Owlv2
from vision_agent_tools.models.florencev2_qa import FlorenceQA


MODEL_REGISTRY: Dict[str, Callable[[], BaseMLModel]] = {
    "florencev2": Florencev2,
    "owlv2": Owlv2,
    "florencev2_qa": FlorenceQA,
}


def get_model_class(model_name: str) -> BaseMLModel:
    """
    Retrieve a model from the registry based on the model name and task

    Args:
        model_name (str): The name of the model to retrieve
        task (Type[Enum]): The enum representing the valid models for a specific task

    Returns:
        BaseMLModel: An instance of the requested model

    Raises:
        ValueError: If the model is not registered or is not valid for the given task
    """

    model_class = MODEL_REGISTRY.get(model_name)

    if not model_class:
        raise ValueError(
            f"Model '{model_name}' is not registered in the model registry."
        )

    return model_class
