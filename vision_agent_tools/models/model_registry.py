from typing import Dict, Callable
from vision_agent_tools.shared_types import BaseMLModel
from vision_agent_tools.models.florencev2 import (
    Florencev2,
)  # Florencev2: interpret simple text prompts to perform tasks like captioning, object detection, and segmentation
from vision_agent_tools.models.owlv2 import Owlv2  # Owlv2: object detection
from vision_agent_tools.models.qr_reader import (
    QRReader,
)  # QRReader: QR code reading
from vision_agent_tools.models.nshot_counting import (
    NShotCounting,
)  # NShotCounting: object counting using the zeroshot and n-shot versions of LOCA model
from vision_agent_tools.models.nsfw_classification import (
    NSFWClassification,
)  # NSFWClassification: classification for NSFW (Not Safe for Work) images
from vision_agent_tools.models.controlnet_aux import Image2Pose
from vision_agent_tools.models.internlm_xcomposer2 import (
    InternLMXComposer2,
)  # InternLMXComposer2: text to image comprehension and composition
from vision_agent_tools.models.clip_media_sim import (
    CLIPMediaSim,
)  # ClipMediaSim: video and target (image or text) to frames that are most similar to the target
from vision_agent_tools.models.depth_anything_v2 import (
    DepthAnythingV2,
)  # DepthAnythingV2: depth estimation


MODEL_REGISTRY: Dict[str, Callable[[], BaseMLModel]] = {
    "florencev2": Florencev2,
    "owlv2": Owlv2,
    "qr_reader": QRReader,
    "nshot_counting": NShotCounting,
    "nsfw_classification": NSFWClassification,
    "image2pose": Image2Pose,
    "internlm_xcomposer2": InternLMXComposer2,
    "clip_media_sim": CLIPMediaSim,
    "depth_anything_v2": DepthAnythingV2,
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
