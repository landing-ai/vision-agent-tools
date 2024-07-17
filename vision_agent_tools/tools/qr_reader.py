import numpy as np
from PIL import Image
from pydantic import BaseModel

from qreader import QReader

from vision_agent_tools.tools.shared_types import (
    BaseTool,
    Device,
    Polygon,
    Point,
    BoundingBox,
)


class QRCodeDetection(BaseModel):
    confidence: float
    text: str
    polygon: Polygon
    bounding_box: BoundingBox
    center: Point


class QRReader(BaseTool):
    def __init__(self):
        self.qreader = QReader()

    def __call__(self, image: Image.Image) -> list[QRCodeDetection]:
        image_array: np.ndarray[np.uint8, np.dtype[np.uint8]] = np.array(image)

        all_text, all_meta = self.qreader.detect_and_decode(
            image=image_array, return_detections=True
        )

        detections = [
            QRCodeDetection(
                confidence=meta["confidence"],
                text=text,
                polygon=Polygon(
                    points=[
                        Point(x=point[0], y=point[1]) for point in meta["polygon_xy"]
                    ]
                ),
                bounding_box=BoundingBox(
                    x_min=meta["bbox_xyxy"][0],
                    y_min=meta["bbox_xyxy"][1],
                    x_max=meta["bbox_xyxy"][2],
                    y_max=meta["bbox_xyxy"][3],
                ),
                center=Point(x=meta["cxcy"][0], y=meta["cxcy"][1]),
            )
            for text, meta in zip(all_text, all_meta)
        ]
        return detections

    def to(self, device: Device):
        self.qreader.detector.model.to(device=device.value)
