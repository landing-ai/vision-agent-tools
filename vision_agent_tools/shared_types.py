from enum import Enum
from typing import Annotated, Literal, TypeVar, Optional, List
from PIL import Image

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    MPS = "mps"


class BaseMLModel:
    """
    Base class for all ML models.
    This class serves as a common interface for all ML models that can be used within tools.
    """

    def predict(
        self, image: Image.Image, prompts: Optional[List[str]] = None, **kwargs
    ):
        """
        Perform a prediction using the model.

        Args:
            image: The input image for prediction.
            prompts: A list of prompts or tasks for the prediction.
            kwargs: Additional model-specific parameters.
        """
        raise NotImplementedError("Subclass must implement 'predict' method")


class BaseTool:
    """
    Base class for all tools that wrap ML models to accomplish tool tasks.
    Tools are responsible for interfacing with one or more ML models to perform specific tasks.
    """

    def __init__(self, model: str):
        self.model = model

    def __call__(self, input, **kwargs):
        raise NotImplementedError("Subclasses should implement '__call__' method.")

    def to(self, device: Device):
        raise NotImplementedError("Subclass must implement 'to' method")


DType = TypeVar("DType", bound=np.generic)

VideoNumpy = Annotated[npt.NDArray[DType], Literal["N", "N", "N", 3]]

SegmentationBitMask = Annotated[npt.NDArray[np.bool_], Literal["N", "N"]]


class Point(BaseModel):
    # X coordinate of the point
    x: float
    # Y coordinate of the point
    y: float


# Class representing a polygon
class Polygon(BaseModel):
    # List of points in the polygon
    points: list[Point]


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
