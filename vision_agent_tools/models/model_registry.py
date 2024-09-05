from typing import Dict, Callable, Type
from vision_agent_tools.shared_types import BaseMLModel

MODELS_PATH = "vision_agent_tools.models"


def lazy_import(model_path: str, class_name: str) -> Type[BaseMLModel]:
    """Lazy import for a model class"""
    module = __import__(model_path, fromlist=[class_name])
    return getattr(module, class_name)


MODEL_REGISTRY: Dict[str, Callable[[], BaseMLModel]] = {
    "florencev2": lambda: lazy_import(f"{MODELS_PATH}.florencev2", "Florencev2"),
    "owlv2": lambda: lazy_import(f"{MODELS_PATH}.owlv2", "Owlv2"),
    "qr_reader": lambda: lazy_import(f"{MODELS_PATH}.qr_reader", "QRReader"),
    "nshot_counting": lambda: lazy_import(
        f"{MODELS_PATH}.nshot_counting", "NShotCounting"
    ),
    "nsfw_classification": lambda: lazy_import(
        f"{MODELS_PATH}.nsfw_classification", "NSFWClassification"
    ),
    "image2pose": lambda: lazy_import(f"{MODELS_PATH}.controlnet_aux", "Image2Pose"),
    "internlm_xcomposer2": lambda: lazy_import(
        f"{MODELS_PATH}.internlm_xcomposer2", "InternLMXComposer2"
    ),
    "clip_media_sim": lambda: lazy_import(
        f"{MODELS_PATH}.clip_media_sim", "CLIPMediaSim"
    ),
    "depth_anything_v2": lambda: lazy_import(
        f"{MODELS_PATH}.depth_anything_v2", "DepthAnythingV2"
    ),
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
