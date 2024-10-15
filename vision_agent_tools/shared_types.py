from enum import Enum
from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from annotated_types import Len
from pydantic import BaseModel, Field


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    MPS = "mps"


class BaseMLModel:
    """
    Base class for all ML models.
    This class serves as a common interface for all ML models that can be used within tools.
    """

    def __call__(self):
        raise NotImplementedError("Subclasses should implement '__call__' method.")


class BaseTool:
    """
    Base class for all tools that wrap ML models to accomplish tool tasks.
    Tools are responsible for interfacing with one or more ML models to perform specific tasks.
    """

    def __call__(self):
        raise NotImplementedError("Subclasses should implement '__call__' method.")


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


# florence2


class PromptTask(str, Enum):
    """Valid task_prompts options for the Florence2 model."""

    CAPTION = "<CAPTION>"
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    OBJECT_DETECTION = "<OD>"
    OCR = "<OCR>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    REGION_TO_SEGMENTATION = "<REGION_TO_SEGMENTATION>"
    REGION_TO_CATEGORY = "<REGION_TO_CATEGORY>"
    REGION_TO_DESCRIPTION = "<REGION_TO_DESCRIPTION>"


class Florence2ModelName(str, Enum):
    FLORENCE_2_LARGE = "microsoft/Florence-2-large"
    FLORENCE_2_BASE_FT = "microsoft/Florence-2-base-ft"


class Florence2TextResponse(BaseModel):
    text: str


class ODResponse(BaseModel):
    labels: list[str]
    bboxes: list[list[float]]


class Florence2OCRResponse(BaseModel):
    labels: list[str]
    quad_boxes: list[list[float]]


class Florence2SegmentationResponse(BaseModel):
    labels: list[str]
    polygons: list[list[list[float]]]


class Florence2OpenVocabularyResponse(BaseModel):
    bboxes: list[list[float]]
    bboxes_labels: list[str]
    polygons: list[list[list[float]]]
    polygons_labels: list[str]


class RLEEncoding(BaseModel):
    counts: list[int]
    size: list[int]


class ImageBboxAndMaskLabel(ODResponse):
    masks: list[RLEEncoding]


# the items can be none for the case where the frame does not have any detections
Florence2ResponseType = (
    list[Florence2TextResponse | None]
    | list[ODResponse | None]
    | list[Florence2OCRResponse | None]
    | list[ImageBboxAndMaskLabel | None]
    | list[Florence2OpenVocabularyResponse | None]
    | list[Florence2SegmentationResponse | None]
)
