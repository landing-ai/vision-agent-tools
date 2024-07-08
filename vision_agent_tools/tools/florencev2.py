from enum import Enum

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from vision_agent_tools.tools.shared_types import BaseTool

MODEL_NAME = "microsoft/Florence-2-large"
PROCESSOR_NAME = "microsoft/Florence-2-large"


class PromptTask(str, Enum):
    OBJECT_DETECTION = "<OD>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"


class Florencev2(BaseTool):
    def __init__(self):
        pass

    def __call__(self, image: Image.Image, prompt: str):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(
            PROCESSOR_NAME, trust_remote_code=True
        )

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )

        return parsed_answer
