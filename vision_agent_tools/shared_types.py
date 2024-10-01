from enum import Enum
from typing import Annotated, Literal, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from annotated_types import Len


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    MPS = "mps"


class BaseMLModel:
    """
    Base class for all ML models.
    This class serves as a common interface for all ML models that can be used within tools.
    """

    def __init__(self, model: str, config: Optional[dict] = None):
        self.model = model

    def __call__(self):
        raise NotImplementedError("Subclasses should implement '__call__' method.")

    def to(self, device: Device):
        raise NotImplementedError("Subclass must implement 'to' method")


class BaseTool:
    """
    Base class for all tools that wrap ML models to accomplish tool tasks.
    Tools are responsible for interfacing with one or more ML models to perform specific tasks.
    """

    def __init__(
        self,
        model: str | BaseMLModel,
    ):
        self.model = model

    def __call__(self):
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


# [x_min, y_min, x_max, y_max] bounding box
BoundingBox = Annotated[list[int | float], Len(min_length=4, max_length=4)]


class BboxLabel(BaseModel):
    label: str
    score: float
    bbox: BoundingBox

    class Config:
        arbitrary_types_allowed = True


class BboxAndMaskLabel(BboxLabel):
    id: int | str
    mask: SegmentationBitMask | None

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True


class FlorenceV2ODRes(BaseModel):
    """
    Schema for the <OD> task.
    """

    bboxes: list[BoundingBox] = Field(
        ..., description="list of bounding boxes, each represented as [x1, y1, x2, y2]"
    )
    labels: list[str] = Field(
        ..., description="list of labels corresponding to each bounding box"
    )

    class Config:
        schema_extra = {
            "example": {
                "<OD>": {
                    "bboxes": [
                        [
                            33.599998474121094,
                            159.59999084472656,
                            596.7999877929688,
                            371.7599792480469,
                        ],
                        [
                            454.0799865722656,
                            96.23999786376953,
                            580.7999877929688,
                            261.8399963378906,
                        ],
                    ],
                    "labels": ["car", "door"],
                }
            }
        }
