from enum import Enum
from pydantic import BaseModel


class Device(str, Enum):
    GPU = "cuda:0"
    CPU = "cpu"


class BaseTool:
    def to(self, device: Device):
        print(device)
        pass


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
