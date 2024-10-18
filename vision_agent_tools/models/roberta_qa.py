import torch
from pydantic import BaseModel
from transformers import pipeline

from vision_agent_tools.shared_types import BaseMLModel, Device

MODEL_NAME = "deepset/roberta-base-squad2"
PROCESSOR_NAME = "deepset/roberta-base-squad2"


class RobertaQAInferenceData(BaseModel):
    """
    Represents an inference result from the Roberta QA model.

    Attributes:
        answer (str): The predicted answer to the question.
        score (float): The confidence score associated with the prediction (between 0 and 1).
    """

    answer: str
    score: float


class RobertaQA(BaseMLModel):
    """
    [Roberta QA](https://huggingface.co/deepset/roberta-base-squad2)
    has been trained on question-answer pairs, including unanswerable questions,
    for the task of Question Answering.

    """

    def __init__(self):
        """
        Initializes the Roberta QA model.
        """
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model = pipeline(
            "question-answering",
            model=MODEL_NAME,
            tokenizer=PROCESSOR_NAME,
            device=self.device,
        )

    @torch.inference_mode()
    def __call__(self, context: str, question: str) -> RobertaQAInferenceData:
        """
        Roberta QA model solves the question-answering task.

        Args:
            context (str): Give context.
            question (str): Give question.

        Returns:
            RobertaQAInferenceData: The output of the Roberta QA model.
                answer (str): The predicted answer to the question.
                score (float): The confidence score associated with the prediction (between 0 and 1).
        """

        with torch.autocast(self.device):
            data = self._model({"context": context, "question": question})
        inference = RobertaQAInferenceData(answer=data["answer"], score=data["score"])
        return inference.model_dump()

    def to(self, device: Device):
        self._model.model.to(device=device.value)
