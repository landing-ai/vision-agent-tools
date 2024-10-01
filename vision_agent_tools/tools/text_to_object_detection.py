import numpy as np
from enum import Enum
from PIL import Image
from pydantic import BaseModel

from vision_agent_tools.models.florencev2 import PromptTask
from vision_agent_tools.models.utils import (
    convert_florence_bboxes_to_bbox_labels,
)
from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.shared_types import (
    BaseTool,
    Device,
    VideoNumpy,
    BboxLabel,
    FlorenceV2ODRes,
)
from vision_agent_tools.models.owlv2 import OWLV2Config


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
        self.modelname: TextToObjectDetectionModel = model

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

    def __call__(
        self,
        prompts: list[str],
        image: Image.Image | None = None,
        video: VideoNumpy[np.uint8] | None = None,
    ) -> list[list[BboxLabel]]:
        """
        Run object detection on the image based on text prompts.

        Args:
            image (Image.Image): The input image for object detection.
            prompts (list[str]): list of text prompts for object detection.

        Returns:
            list[list[BboxLabel]]: A list of detection results for each prompt.
        """
        prediction: list[list[BboxLabel]] = []
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if self.modelname == TextToObjectDetectionModel.OWLV2:
            if image is not None:
                prediction = self.model(image=image, prompts=prompts)
            elif video is not None:
                prediction = self.model(video=video, prompts=prompts)

        elif self.modelname == TextToObjectDetectionModel.FLORENCEV2:
            od_task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
            prompt = ", ".join(prompts)
            if image is not None:
                fl_prediction = self.model(
                    image=image,
                    task=od_task,
                    prompt=prompt,
                )
                fl_prediction = [FlorenceV2ODRes(**fl_prediction[od_task])]
            elif video is not None:
                fl_prediction = self.model(
                    video=video,
                    task=od_task,
                    prompt=prompt,
                )
                fl_prediction = [
                    FlorenceV2ODRes(**pred[od_task]) for pred in fl_prediction
                ]
            # Prediction should be a list of lists of BboxLabel objects
            # We need to convert the output to the format that is used in the playground-tools
            fv2_pred_output = []
            for pred in fl_prediction:
                fv2_pred_output.append(convert_florence_bboxes_to_bbox_labels(pred))
            prediction = fv2_pred_output
        return prediction

    def to(self, device: Device):
        self.model.to(device)
        return self
