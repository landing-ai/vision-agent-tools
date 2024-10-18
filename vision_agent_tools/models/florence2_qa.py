from typing import Any

import torch
from PIL import Image

from vision_agent_tools.shared_types import PromptTask
from vision_agent_tools.models.florence2 import Florence2
from vision_agent_tools.models.roberta_qa import RobertaQA
from vision_agent_tools.shared_types import BaseMLModel, Device


class FlorenceQA(BaseMLModel):
    """
    FlorenceQA is a tool that combines the Florence-2 and Roberta QA models
    to answer questions about images.

    NOTE: The Florence-2 model can only be used in GPU environments.
    """

    def __init__(self) -> None:
        """
        Initializes the FlorenceQA model.
        """
        self._florence = Florence2()
        self._roberta_qa = RobertaQA()

    @torch.inference_mode()
    def __call__(self, image: Image.Image, question: str) -> dict[str, Any]:
        """
        FlorenceQA model answers questions about images.

        Args:
            image (Image.Image): The image to be analyzed.
            question (str): The question to be answered.

        Returns:
            str: The answer to the question.
        """
        image = image.convert("RGB")
        task = PromptTask.MORE_DETAILED_CAPTION
        output_caption = self._florence(images=[image], task=task)[0]
        roberta_pred = self._roberta_qa(output_caption["text"], question)
        return {"text": roberta_pred["answer"]}

    def to(self, device: Device):
        raise NotImplementedError("This method is not supported for FlorenceQA model.")
