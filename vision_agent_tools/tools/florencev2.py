import os
import torch

from typing import Optional, Any
from enum import Enum
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor
from vision_agent_tools.shared_types import BaseTool
from pathlib import Path
from vision_agent_tools.shared_types import DEFAULT_HF_CHACHE_DIR
from huggingface_hub import snapshot_download


class PromptTask(str, Enum):
    """
    Valid task_prompts options for the Florence2 model.

    """

    CAPTION = "<CAPTION>"
    """"""
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    """"""
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    """"""
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    """"""
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    """"""
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """"""
    OBJECT_DETECTION = "<OD>"
    """"""
    OCR = "<OCR>"
    """"""
    OCR_WITH_REGION = "<OCR_WITH_REGION>"
    """"""
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    """"""
    REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    """"""
    REGION_TO_SEGMENTATION = "<REGION_TO_SEGMENTATION>"
    """"""
    REGION_TO_CATEGORY = "<REGION_TO_CATEGORY>"
    """"""
    REGION_TO_DESCRIPTION = "<REGION_TO_DESCRIPTION>"
    """"""


class Florencev2(BaseTool):
    """
    [Florence-2](https://huggingface.co/microsoft/Florence-2-base) can interpret simple
    text prompts to perform tasks like captioning, object detection, and segmentation.

    NOTE: The Florence-2 model can only be used in GPU environments.
    """

    _MODEL_NAME = "microsoft/Florence-2-large"

    def __init__(self, cache_dir: str | Path | None = None):
        """
        Initializes the Florence-2 model.
        """
        model_dir = (
            f"models--{os.path.dirname(self._MODEL_NAME)}"
            + f"--{os.path.basename(self._MODEL_NAME)}"
        )
        default_cache_model_dir = os.path.join(DEFAULT_HF_CHACHE_DIR, model_dir)

        user_cached_folder = (
            os.path.join(cache_dir, model_dir) if cache_dir is not None else ""
        )

        is_user_cached_folder = True if os.path.exists(user_cached_folder) else False
        is_default_cached_folder = (
            True if os.path.exists(default_cache_model_dir) else False
        )
        is_model_cached = is_user_cached_folder or is_default_cached_folder
        print("Is the model cached?:  ", is_model_cached)
        print(
            "Using cache folder: ",
            user_cached_folder if is_user_cached_folder else default_cache_model_dir,
        )
        model_snapshot = snapshot_download(
            self._MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=is_model_cached,
        )
        print("Model_snapshot_path: ", model_snapshot)
        # If there is no cache_dir provided then, the default store path is:
        # /root/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/6bf179230dxxx
        self._model = AutoModelForCausalLM.from_pretrained(
            model_snapshot, trust_remote_code=True, local_files_only=True
        )
        self._processor = AutoProcessor.from_pretrained(
            model_snapshot, trust_remote_code=True, local_files_only=True
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def __call__(
        self, image: Image.Image, task: PromptTask, prompt: Optional[str] = ""
    ) -> Any:
        """
        Florence-2 model sequence-to-sequence architecture enables it to excel in both
        zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.
        For more examples and details, refer to the [Florence-2 sample usage](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb).

        Args:
            image (Image.Image): The input image for object detection.
            task (PromptTask): The task to be performed on the image.
            prompt (Optional[str]): The text input that complements the prompt task.

        Returns:
            Any: The output of the Florence-2 model based on the task and prompt.
        """
        if prompt is None:
            text_input = task
        else:
            text_input = task + prompt

        image = image.convert("RGB")

        inputs = self._processor(text=text_input, images=image, return_tensors="pt").to(
            self.device
        )

        with torch.autocast(self.device):
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=False,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self._processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )

        return parsed_answer
