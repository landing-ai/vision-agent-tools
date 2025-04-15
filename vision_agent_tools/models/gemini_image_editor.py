from typing import List
from PIL import Image
from pydantic import Field
from pydantic import ConfigDict, validate_call
from vision_agent_tools.shared_types import BaseMLModel, Device
from google import genai
from google.genai import types
import numpy as np
import base64
from io import BytesIO


def b64_to_pil(b64_str: str) -> Image.Image:
    # , can't be encoded in b64 data so it must be part of prefix
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(b64_str)))


class GeminiImageEditor(BaseMLModel):
    """
    Tool for image editing using the Gemini 2.0 Flash model.
    This tool takes a prompt and an image as input and generates 
    an image using the Gemini 2.0 Flash model.
    """

    config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model_name: str | None = "gemini-2.0-flash-exp-image-generation",
    ):
        """
        Initializes the Gemini image generation tool.

        Args:
            - model_name (str): The name of the image generation model to use.
                Currently, only "gemini-2.0-flash-exp-image-generation" is supported.
        """
        self._model = None

        if model_name == "gemini-2.0-flash-exp-image-generation":
            self._model = model_name
        else:
            raise ValueError(
                f"Unsupported model name: {model_name}. Supported models are: gemini-2.0-flash-exp-image-generation."
            )


    @validate_call(config=config)
    def __call__(
        self,
        prompt: str = Field(max_length=512),
        image: Image.Image | None = None,
    ) -> Image.Image | None:
        """
        Performs image editing using the Gemini model and a prompt.

        Args:
            - prompt (str): The text prompt describing the desired modifications.
            - image (Image.Image): The original image to be modified.

        Returns:
            Image.Image: Generated image if successful; None if an error occurred.
        """
        output = None
        client = genai.Client()

        input = [prompt, image] if image else [prompt]
        response = client.models.generate_content(
            model=self._model,
            contents=input,
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
        )

        candidate = response.candidates[0]
        part = candidate.content.parts[0]
        inline_data = part.inline_data
        data = inline_data.data

        if not data or data == b"":
            raise ValueError("Inline data is empty or missing in the response.")

        try:
            output = np.array(b64_to_pil(data.decode("utf-8")))
        except UnicodeDecodeError:
            output = np.array(Image.open(BytesIO(data)))

        return output

    def to(self, device: Device):
        raise NotImplementedError("This method is not supported for Gemini Image Editing.")