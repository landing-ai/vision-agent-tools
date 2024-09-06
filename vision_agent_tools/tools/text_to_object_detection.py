from typing import List, Any
from enum import Enum

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from vision_agent_tools.models.florencev2 import FlorenceV2ODOutput, PromptTask
from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.shared_types import BaseTool, VideoNumpy

from vision_agent_tools.models.owlv2 import OWLV2Config
from typing import Annotated, Any, List, Optional
from annotated_types import Len

BoundingBox = Annotated[list[int | float], Len(min_length=4, max_length=4)]


class ODResponseData(BaseModel):
    label: str
    score: float
    bbox: BoundingBox = Field(alias="bounding_box")

    model_config = ConfigDict(
        populate_by_name=True,
    )


class TextToObjectDetectionOutput(BaseModel):
    """This can be a list of lists of ODResponseData objects,
    each list can be the frame of a video or the image of a batch of images,
    then inside the list it is the list of ODResponseData objects for each object detected in the frame or image.

    Downstream usage example, later the playground-tools (baseten model APIs) can wrap this 2-d list in the datafield of BaseReponse.
    """

    output: list[list[ODResponseData]]


class TextToObjectDetectionModel(str, Enum):
    OWLV2 = "owlv2"
    FLORENCEV2 = "florencev2"


class TextToObjectDetection(BaseTool):
    """
    Tool to perform object detection based on text prompts using a specified ML model
    """

    def __init__(
        self,
        model: TextToObjectDetectionModel,
        model_config: OWLV2Config | None = None,
    ):
        if model not in TextToObjectDetectionModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )

        if model == TextToObjectDetectionModel.OWLV2:
            self.owlv2_config = model_config or OWLV2Config()
        elif model == TextToObjectDetectionModel.FLORENCEV2:
            pass  # Note we don't need model config for FlorenceV2
        else:
            raise ValueError(f"Model is not supported: '{model}'")

        self.model_class = get_model_class(model_name=model)
        model_instance = self.model_class()
        if model == TextToObjectDetectionModel.OWLV2:
            super().__init__(model=model_instance(self.owlv2_config))
        elif model == TextToObjectDetectionModel.FLORENCEV2:
            super().__init__(model=model_instance())

    def _convert_florencev2_output(
        self, output: FlorenceV2ODOutput
    ) -> list[ODResponseData]:
        """

        Args:
            output (example):
            {'<OD>': {'bboxes': [[33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]], 'labels': ['car', 'door', 'door', 'wheel', 'wheel']}}

        Returns:
            ODResponse: the one we are using already in the playground-tools(baseten) ,that will be wrapped in the datafield of BaseReponse.
        """
        od_response = []
        for i, label in enumerate(output.labels):
            od_response.append(
                ODResponseData(
                    label=label,
                    score=1.0,  # FlorenceV2 doesn't provide confidence score
                    bbox=output.bboxes[i],
                )
            )

    def __call__(
        self,
        prompts: List[str],
        image: Image.Image | None = None,
        video: VideoNumpy[np.uint8] | None = None,
    ) -> List[TextToObjectDetectionOutput]:
        """
        Run object detection on the image based on text prompts.

        Args:
            image (Image.Image): The input image for object detection.
            prompts (List[str]): List of text prompts for object detection.

        Returns:
            List[TextToObjectDetectionOutput]: A list of detection results for each prompt.
        """
        results = []

        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            prediction = self.model(image=image, prompts=prompts)
        if video is not None:
            prediction = self.model(video=video, prompts=prompts)

        output = TextToObjectDetectionOutput(output=prediction)
        results.append(output)

        return results
        # results = []
        # prediction: list[list[ODResponseData]] = []
        # if image is None and video is None:
        #     raise ValueError("Either 'image' or 'video' must be provided.")
        # if image is not None and video is not None:
        #     raise ValueError("Only one of 'image' or 'video' can be provided.")

        # if self.model == TextToObjectDetectionModel.OWLV2:
        #     if image is not None:
        #         prediction = self.model(image=image, prompts=prompts)
        #     elif video is not None:
        #         prediction = self.model(video=video, prompts=prompts)

        # elif self.model == TextToObjectDetectionModel.FLORENCEV2:
        #     od_task = PromptTask.OBJECT_DETECTION
        #     if image is not None:
        #         prediction = self.model(
        #             image=image,
        #             task=od_task,
        #         )
        #     elif video is not None:
        #         prediction = self.model(
        #             video=video,
        #             task=od_task,
        #         )

        #     # Prediction should be a list of lists of ODResponseData objects
        #     # We need to convert the output to the format that is used in the playground-tools
        #     for pred in prediction:
        #         prediction.append(self._convert_florencev2_output(pred[od_task]))

        # output = TextToObjectDetectionOutput(output=prediction)
        # results.append(output)

        # return results
