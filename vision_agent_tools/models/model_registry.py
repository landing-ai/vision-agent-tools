import re
from typing import Dict, Type

from pydantic import BaseModel, field_validator

from vision_agent_tools.shared_types import BaseMLModel

MODELS_PATH = "vision_agent_tools.models"


class ModelRegistryEntry(BaseModel):
    model_name: str
    class_name: str

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model names are lowercase and separated by underscores."""
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"Model name '{v}' must be lowercase and separated by underscores."
            )
        return v

    def model_import(self) -> Type[BaseMLModel]:
        """Lazy import for a model class."""
        module = __import__(
            f"{MODELS_PATH}.{self.model_name}", fromlist=[self.class_name]
        )
        return getattr(module, self.class_name)


MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {
    "florence2": ModelRegistryEntry(
        model_name="florence2",
        class_name="Florence2",
    ),
    "florence2sam2": ModelRegistryEntry(
        model_name="florence2_sam2",
        class_name="Florence2SAM2",
    ),
    "owlv2": ModelRegistryEntry(model_name="owlv2", class_name="Owlv2"),
    "qr_reader": ModelRegistryEntry(
        model_name="qr_reader",
        class_name="QRReader",
    ),
    "nshot_counting": ModelRegistryEntry(
        model_name="nshot_counting",
        class_name="NShotCounting",
    ),
    "nsfw_classification": ModelRegistryEntry(
        model_name="nsfw_classification",
        class_name="NSFWClassification",
    ),
    "image2pose": ModelRegistryEntry(
        model_name="image2pose",
        class_name="Image2Pose",
    ),
    "internlm_xcomposer2": ModelRegistryEntry(
        model_name="internlm_xcomposer2",
        class_name="InternLMXComposer2",
    ),
    "clip_media_sim": ModelRegistryEntry(
        model_name="clip_media_sim",
        class_name="CLIPMediaSim",
    ),
    "depth_anything_v2": ModelRegistryEntry(
        model_name="depth_anything_v2",
        class_name="DepthAnythingV2",
    ),
    "flux1": ModelRegistryEntry(model_name="flux1", class_name="Flux1"),
    "qwen2_vl": ModelRegistryEntry(
        model_name="qwen2_vl",
        class_name="Qwen2VL",
    ),
    "siglip": ModelRegistryEntry(model_name="siglip", class_name="Siglip"),
}


def get_model_class(model_name: str) -> BaseMLModel:
    """
    Retrieve a model from the registry based on the model name and task

    Args:
        model_name (str): The name of the model to retrieve

    Returns:
        BaseMLModel: An instance of the requested model

    Raises:
        ValueError: If the model is not registered.
    """

    entry = MODEL_REGISTRY.get(model_name)

    if not entry:
        raise ValueError(
            f"Model '{model_name}' is not registered in the model registry."
        )

    return entry.model_import
