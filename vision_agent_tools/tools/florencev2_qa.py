import torch
from PIL import Image

from vision_agent_tools.helpers.roberta_qa import RobertaQA
from vision_agent_tools.shared_types import BaseTool, Device
from vision_agent_tools.tools.florencev2 import Florencev2, PromptTask


class FlorenceQA(BaseTool):
    """
    FlorenceQA is a tool that combines the Florence-2 and Roberta QA models
    to answer questions about images.

    NOTE: The Florence-2 model can only be used in GPU environments.
    """

    def __init__(self) -> None:
        """
        Initializes the FlorenceQA model.
        """
        self._florence = Florencev2()
        self._roberta_qa = RobertaQA()

    @torch.inference_mode()
    def __call__(self, image: Image.Image, question: str) -> str:
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
        output_caption = self._florence(image, task)
        caption = output_caption[task]
        answer = self._roberta_qa(caption, question)

        return answer.answer

    def to(self, device: Device):
        self._florence.to(device=device)
        self._roberta_qa.to(device=device)
