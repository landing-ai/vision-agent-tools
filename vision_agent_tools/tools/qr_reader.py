from enum import Enum
from PIL import Image
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool
from vision_agent_tools.models.model_registry import get_model_class


class QRReaderOutput(BaseModel):
    text: str


class QRReaderModel(str, Enum):
    QR_READER = "qr_reader"


class QRReader(BaseTool):
    """
    Tool to perform QR reading on an image.
    """

    def __init__(self, model: QRReaderModel):
        if model not in QRReaderModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )
        model_class = get_model_class(model_name=model)
        model_instance = model_class()
        super().__init__(model=model_instance())

    def __call__(self, image: Image.Image) -> QRReaderOutput:
        result = self.model(image=image)

        if isinstance(result, list) and len(result) > 0:
            detected_text = result[0].text
        else:
            detected_text = ""

        return QRReaderOutput(text=detected_text)
