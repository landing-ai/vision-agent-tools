from enum import Enum
from PIL import Image
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class


class TextToClassificationOutput(BaseModel):
    label: str
    score: float


class TextToClassificationModel(str, Enum):
    NSFW_CLASSIFICATION = "nsfw_classification"


class TextToClassification(BaseTool):
    """
    Tool to classify images based on text prompts.
    """

    def __init__(self, model: TextToClassificationModel):
        if model not in TextToClassificationModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )
        model_class = get_model_class(model_name=model)
        model_instance = model_class()
        super().__init__(model=model_instance())

    def __call__(self, image: Image.Image) -> list[TextToClassificationOutput]:
        results = self.model(image=image)
        return TextToClassificationOutput(label=results.label, score=results.score)
