import torch

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pydantic import BaseModel
from vision_agent_tools.shared_types import BaseTool, CachePath
from vision_agent_tools.tools.utils import manage_hf_model_cache


class RobertaQAInferenceData(BaseModel):
    """
    Represents an inference result from the Roberta QA model.

    Attributes:
        answer (str): The predicted answer to the question.
        score (float): The confidence score associated with the prediction (between 0 and 1).
    """

    answer: str
    score: float


class RobertaQA(BaseTool):
    """
    [Roberta QA](https://huggingface.co/deepset/roberta-base-squad2)
    has been trained on question-answer pairs, including unanswerable questions,
    for the task of Question Answering.

    """

    _MODEL_NAME = "deepset/roberta-base-squad2"

    def __init__(self, cache_dir: CachePath = None):
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
        model_snapshot = manage_hf_model_cache(self._MODEL_NAME, cache_dir)
        self._model = AutoModelForQuestionAnswering.from_pretrained(
            model_snapshot, trust_remote_code=True, local_files_only=True
        )
        self._processor = AutoTokenizer.from_pretrained(
            model_snapshot, trust_remote_code=True, local_files_only=True
        )

        self._pipeline = pipeline(
            "question-answering",
            model=self._model,
            tokenizer=self._processor,
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
            data = self._pipeline({"context": context, "question": question})
        inference = RobertaQAInferenceData(answer=data["answer"], score=data["score"])

        return inference
