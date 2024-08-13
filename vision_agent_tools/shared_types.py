from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel
import numpy as np
import numpy.typing as npt


class BaseTool:
    pass


DType = TypeVar("DType", bound=np.generic)

VideoNumpy = Annotated[npt.NDArray[DType], Literal["N", "N", "N", 3]]

SegmentationMask = Annotated[npt.NDArray[np.bool_], Literal["N", "N"]]


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
