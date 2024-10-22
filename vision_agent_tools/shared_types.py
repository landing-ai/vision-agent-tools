from enum import Enum
from typing import Annotated, Literal, TypeVar, Any

import numpy as np
import numpy.typing as npt
from annotated_types import Len
from pydantic import BaseModel, Field, ConfigDict, field_serializer


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    MPS = "mps"


class BaseMLModel:
    """
    Base class for all ML models.
    This class serves as a common interface for all ML models that can be used within tools.
    """

    def __init__(self, model: str, config: dict[str, Any] | None = None):
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
    labels: list[str] = Field(
        description="list of labels corresponding to each bounding box"
    )
    bboxes: list[list[float]] = Field(
        description="list of bounding boxes, each represented as [x_min, y_min, x_max, y_max]"
    )


class ODWithScoreResponse(ODResponse):
    scores: list[float] = Field(
        description="list of confidence scores for each bounding box"
    )


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


# the items can be none for the case where the frame does not have any detections
Florence2ResponseType = (
    list[Florence2TextResponse | None]
    | list[ODResponse | None]
    | list[Florence2OCRResponse | None]
    | list[Florence2OpenVocabularyResponse | None]
    | list[Florence2SegmentationResponse | None]
)


# sam2

SegmentationBitMask = Annotated[npt.NDArray[np.bool_], Literal["H", "W"]]


class RLEEncoding(BaseModel):
    counts: list[int]
    size: list[int]


class Sam2BitMask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    masks: list[SegmentationBitMask]
    scores: list[float]
    logits: list[SegmentationBitMask]


class BboxAndMaskLabel(ODResponse):
    masks: list[RLEEncoding]


class ObjBboxAndMaskLabel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    label: str
    bbox: list[float]
    mask: SegmentationBitMask

    @field_serializer('mask')
    def serialize_mask(self, mask: SegmentationBitMask, _info):
        return _binary_mask_to_rle(mask)


class ObjMaskLabel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    score: float | None
    logits: SegmentationBitMask | None
    mask: SegmentationBitMask

    @field_serializer('mask')
    def serialize_mask(self, mask: SegmentationBitMask, _info):
        return _binary_mask_to_rle(mask)


def _binary_mask_to_rle(binary_mask: np.ndarray) -> RLEEncoding:
    counts = []
    size = list(binary_mask.shape)

    flattened_mask = binary_mask.ravel(order="F")
    nonzero_indices = np.flatnonzero(flattened_mask[1:] != flattened_mask[:-1]) + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    if flattened_mask[0] == 1:
        lengths = np.insert(lengths, 0, 0)

    counts = lengths.tolist()
    return RLEEncoding(counts=counts, size=list(size))
