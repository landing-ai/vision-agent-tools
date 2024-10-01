import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from qreader import QReader

from vision_agent_tools.shared_types import (
    BaseMLModel,
    BoundingBox,
    Device,
    Point,
    Polygon,
)


class QRCodeDetection(BaseModel):
    """
    Represents a detected QR code.
    """

    confidence: float = Field(
        description="The confidence score associated with the detection (between 0 and 1)"
    )
    text: str = Field(description="The decoded text content of the QR code")
    polygon: Polygon = Field(
        description="A `Polygon` object representing the detected QR code's corner points"
    )
    bbox: BoundingBox = Field(
        description="A `BoundingBox` object representing the axis-aligned bounding box coordinates of the detected QR code"
    )
    center: Point = Field(
        description="A `Point` object representing the center coordinates of the detected QR code"
    )


class QRReader(BaseMLModel):
    """
    This tool utilizes the `qreader` library to detect QR codes within an input image.
    It returns a list of `QRCodeDetection` objects for each detected QR code, containing
    the decoded text, confidence score, polygon coordinates, bounding box, and center point.
    """

    def __init__(self):
        """
        Initializes the QR code reader tool.

        Loads the `QReader` instance for QR code detection.
        """

        self.qreader = QReader()

    def __call__(self, image: Image.Image) -> list[QRCodeDetection]:
        """
        Detects QR codes in an image.

        Args:
            image (Image.Image): The input image for QR code detection.

        Returns:
            list[QRCodeDetection]: A list of `QRCodeDetection` objects containing
                                   information about each detected QR code, or an empty list if none are found.
        """

        image_array: np.ndarray[np.uint8, np.dtype[np.uint8]] = np.array(image)

        all_text, all_meta = self.qreader.detect_and_decode(
            image=image_array, return_detections=True
        )

        detections = [
            QRCodeDetection(
                confidence=meta["confidence"],
                text=text,
                polygon=Polygon(
                    points=[Point(x=point[0], y=point[1]) for point in meta["quad_xy"]]
                ),
                bbox=[
                    meta["bbox_xyxy"][0],  # x_min
                    meta["bbox_xyxy"][1],  # y_min
                    meta["bbox_xyxy"][2],  # x_max
                    meta["bbox_xyxy"][3],  # y_max
                ],
                center=Point(x=meta["cxcy"][0], y=meta["cxcy"][1]),
            )
            for text, meta in zip(all_text, all_meta)
        ]
        return detections

    def to(self, device: Device):
        self.qreader.detector.model.to(device=device.value)
