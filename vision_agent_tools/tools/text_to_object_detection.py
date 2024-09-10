from typing import List, Any
from enum import Enum

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from vision_agent_tools.models.florencev2 import FlorenceV2ODRes, PromptTask
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


class TextToObjectDetectionRes(BaseModel):
    """This can be a list of lists of ODResponseData objects,
    each list can be the frame of a video or the image of a batch of images,
    then inside the list it is the list of ODResponseData objects for each object detected in the frame or image.

    Downstream usage example, later the playground-tools (baseten model APIs) can wrap this 2-d list in the datafield of BaseReponse.
    """

    res: list[list[ODResponseData]]


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

        # Later modal is changed to actual model object
        self.modelname:TextToObjectDetectionModel = model

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

    def _convert_florencev2_res(
        self, res: FlorenceV2ODRes
    ) -> list[ODResponseData]:
        """

        Returns:
            ODResponse: the one we are using already in the playground-tools(baseten) ,that will be wrapped in the datafield of BaseReponse.
        """
        print(f"converting res: {res}")
        od_response = []
        for i, label in enumerate(res.labels):
            od_response.append(
                ODResponseData(
                    label=label,
                    score=1.0,  # FlorenceV2 doesn't provide confidence score
                    bbox=res.bboxes[i],
                )
            )

    def _convert_owlv2_res(
        self, res: list[list[ODResponseData]]
    ) -> list[ODResponseData]:
        """
        Convert the output of the OWLV2 model to the format used in the playground-tools.

        Args:
            output (list[Owlv2InferenceData]): The output from the OWLV2 model.
        class Owlv2InferenceData(BaseModel):
            label: str = Field(description="The predicted label for the detected object")
            score: float = Field(
                description="TThe confidence score associated with the prediction (between 0 and 1)"
            )
            bbox: list[float] = Field(
                description=" A list of four floats representing the bounding box coordinates (xmin, ymin, xmax, ymax) of the detected object in the image"
            )


        Returns:
            list[ODResponseData]: The converted output.
        """
        od_response = []
        for pred in res:
            single_img_od_response = []
            od_response.append(single_img_od_response)
            for box in pred:
                single_img_od_response.append(
                    ODResponseData(
                        label=box.label,
                        score=box.score,
                        bbox=box.bbox,
                    )
                )

        return od_response

    def __call__(
        self,
        prompts: List[str],
        image: Image.Image | None = None,
        video: VideoNumpy[np.uint8] | None = None,
    ) -> List[TextToObjectDetectionRes]:
        """
        Run object detection on the image based on text prompts.

        Args:
            image (Image.Image): The input image for object detection.
            prompts (List[str]): List of text prompts for object detection.

        Returns:
            List[TextToObjectDetectionOutput]: A list of detection results for each prompt.
        """
        # results = []

        # if image is None and video is None:
        #     raise ValueError("Either 'image' or 'video' must be provided.")
        # if image is not None and video is not None:
        #     raise ValueError("Only one of 'image' or 'video' can be provided.")

        # if image is not None:
        #     prediction = self.model(image=image, prompts=prompts)
        # if video is not None:
        #     prediction = self.model(video=video, prompts=prompts)

        # output = TextToObjectDetectionOutput(
        #     output=self._convert_owlv2_output(prediction)
        # )
        # results.append(output)

        # return results

        results = []
        prediction: list[list[ODResponseData]] = []
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if self.modelname == TextToObjectDetectionModel.OWLV2:
            if image is not None:
                prediction = self.model(image=image, prompts=prompts)
            elif video is not None:
                prediction = self.model(video=video, prompts=prompts)

            prediction = self._convert_owlv2_res(prediction)

        elif self.modelname== TextToObjectDetectionModel.FLORENCEV2:
            od_task = PromptTask.OBJECT_DETECTION
            if image is not None:
                prediction = self.model(
                    image=image,
                    task=od_task,
                )
            elif video is not None:
                prediction = self.model(
                    video=video,
                    task=od_task,
                )

            # Prediction should be a list of lists of ODResponseData objects
            # We need to convert the output to the format that is used in the playground-tools
            fv2_pred_output=[]
            if not isinstance(prediction, list):
                prediction = [prediction]
            for pred in prediction:
                print(f"each of pred is {pred}, type: {type(pred)}")
                fv2_pred_output.append(self._convert_florencev2_res(FlorenceV2ODRes(**pred[od_task])))
                print(f"each of fv2_pred_output is {fv2_pred_output}, type: {type(fv2_pred_output)}")
            prediction = fv2_pred_output

        print(f"prediction for TextToObjectDetectionRes is {prediction}")
        res = TextToObjectDetectionRes(res=prediction)
        results.append(res)

        return results
