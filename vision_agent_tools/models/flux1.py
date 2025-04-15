import random
from enum import Enum
from typing import List, Annotated
import torch
from PIL import Image
from pydantic import BaseModel, Field
from pydantic import ConfigDict, validate_arguments
from pydantic.functional_validators import AfterValidator
from diffusers import FluxPipeline, FluxInpaintPipeline, FluxImg2ImgPipeline

from vision_agent_tools.shared_types import BaseMLModel, Device


class Flux1Task(str, Enum):
    IMAGE_GENERATION = "generation"
    MASK_INPAINTING = "inpainting"
    IMAGE_TO_IMAGE = "img2img"


def _check_multiple_of_8(number: int) -> int:
    assert number % 8 == 0, "height and width must be multiples of 8."
    return number


class Flux1Config(BaseModel):
    """
    Configuration for the Flux1 model.
    """

    height: Annotated[int, AfterValidator(_check_multiple_of_8)] = Field(
        ge=8, default=512
    )
    width: Annotated[int, AfterValidator(_check_multiple_of_8)] = Field(
        ge=8, default=512
    )
    num_inference_steps: int | None = Field(ge=1, default=10)
    guidance_scale: float | None = Field(ge=0, default=3.5)
    num_images_per_prompt: int | None = Field(ge=1, default=1)
    max_sequence_length: int | None = Field(ge=0, le=512, default=512)
    seed: int | None = None
    strength: float | None = Field(ge=0, le=1, default=0.6)


class Flux1(BaseMLModel):
    """
    Tool for object detection using the pre-trained Flux1 model.
    This tool takes a prompt as input and generates an image using the Flux1 model.
    """

    config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        hf_model: str = "black-forest-labs/FLUX.1-schnell",
        dtype: torch.dtype = torch.bfloat16,
        enable_sequential_cpu_offload: bool = True,
        hf_access_token: str | None = None,
    ):
        """
        Initializes the Flux1 image generation tool.
        Loads the pre-trained Flux1 model from HuggingFace and sets model configurations.

        Args:
            - task (Flux1Task): The task to perform using the model:
                either image generation ("generation")
                or mask inpainting ("inpainting").
            - model_config: The configuration for the model, hf_model, and device.
            - dtype (torch.dtype): The data type to use for the model.
            - enable_sequential_cpu_offload (bool): Whether to enable sequential CPU offload.
        """

        self._pipeline_img_generation = FluxPipeline.from_pretrained(
            hf_model, torch_dtype=dtype, token=hf_access_token
        )
        if enable_sequential_cpu_offload:
            self._pipeline_img_generation.enable_sequential_cpu_offload()

        self._pipeline_mask_inpainting = FluxInpaintPipeline.from_pretrained(
            hf_model, torch_dtype=dtype, token=hf_access_token
        )
        if enable_sequential_cpu_offload:
            self._pipeline_mask_inpainting.enable_sequential_cpu_offload()

        self._pipeline_img2img = FluxImg2ImgPipeline.from_pretrained(
            hf_model, torch_dtype=dtype, token=hf_access_token
        )
        if enable_sequential_cpu_offload:
            self._pipeline_img2img.enable_sequential_cpu_offload()

    @torch.inference_mode()
    @validate_arguments(config=config)
    def __call__(
        self,
        prompt: str = Field(max_length=512),
        task: Flux1Task = Flux1Task.IMAGE_GENERATION,
        config: Flux1Config = Flux1Config(),
        image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
    ) -> List[Image.Image] | None:
        """
        Performs object detection on an image using the Flux1 model.

        Args:
            - prompt (str): The text prompt describing the desired modifications.
            - task (Flux1Task): The task to perform using the model:
                - image generation - "generation",
                - mask inpainting - "inpainting",
                - image-to-image generation - "img2img".
            - config (Flux1Config):
                - height (`int`, *optional*):
                    The height in pixels of the generated image.
                    This is set to 512 by default.
                - width (`int`, *optional*):
                    The width in pixels of the generated image.
                    This is set to 512 by default.
                - num_inference_steps (`int`, *optional*, defaults to 28):
                - guidance_scale (`float`, *optional*, defaults to 3.5):
                    Guidance scale as defined in Classifier-Free Diffusion Guidance.
                    Higher guidance scale encourages to generate images
                    that are closely linked to the text `prompt`,
                    usually at the expense of lower image quality.
                - num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                - max_sequence_length (`int` defaults to 512):
                    Maximum sequence length to use with the `prompt`.
                    to make generation deterministic.
                - seed (`int`, *optional*): The seed to use for the random number generator.
                    If not provided, a random seed is used.
                - strength (`float`, *optional*, defaults to 0.6):
                    Indicates extent to transform the reference `image`.
                    Must be between 0 and 1.
                    A value of 1 essentially ignores `image`.
            - image (Image.Image): The original image to be modified.
            - mask_image (Image.Image): The mask image indicating areas to be inpainted.

        Returns:
            List[Image.Image]: The list of generated image(s) if successful; None if an error occurred.
        """

        seed = config.seed

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator("cpu").manual_seed(seed)
        output = None

        if task == Flux1Task.IMAGE_GENERATION:
            output = self._generate_image(
                prompt=prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=config.num_images_per_prompt,
                max_sequence_length=config.max_sequence_length,
                generator=generator,
            )
        elif task == Flux1Task.MASK_INPAINTING:
            if image is None or mask_image is None:
                raise ValueError(
                    "Both image and mask image must be provided for inpainting."
                )

            if image.size != mask_image.size:
                raise ValueError("The image and mask image should have the same size.")

            height, width = config.height, config.width

            if height is None or width is None:
                height, width = image.size

            output = self._inpaint_image(
                image=image,
                mask_image=mask_image,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=config.num_images_per_prompt,
                max_sequence_length=config.max_sequence_length,
                strength=config.strength,
                generator=generator,
            )
        elif task == Flux1Task.IMAGE_TO_IMAGE:
            if image is None:
                raise ValueError(
                    "Image must be provided for image-to-image generation."
                )

            height, width = config.height, config.width

            if height is None or width is None:
                height, width = image.size

            output = self._image_to_image(
                prompt=prompt,
                image=image,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=config.num_images_per_prompt,
                max_sequence_length=config.max_sequence_length,
                generator=generator,
            )
        else:
            raise ValueError(
                f"Unsupported task: {task}. Supported tasks are: {', '.join([task.value for task in Flux1Task])}."
            )

        return output

    def to(self, device: Device):
        raise NotImplementedError("This method is not supported for Flux1 model.")

    def _generate_image(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        generator: torch.Generator,
        max_sequence_length: int,
    ) -> List[Image.Image] | None:
        """
        Generate an image from a given prompt.

        Image generation pipeline to create an image based on a provided textual prompt.

        Args:
            prompt (`str`)
            height (`int`)
            width (`int`)
            num_inference_steps (`int`)
            guidance_scale (`float`)
            num_images_per_prompt (`int`)
            generator (`torch.Generator`)
            max_sequence_length (`int`)

        Returns:
            List[Image.Image]: The list of generated image(s) if successful; None if an error occurred.
        """
        output = self._pipeline_img_generation(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )

        if output is None:
            return None

        return output.images

    def _inpaint_image(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        generator: torch.Generator,
        max_sequence_length: int,
        strength: float,
    ) -> List[Image.Image] | None:
        """
        Inpaint an image using a given prompt and a mask image.

        This method utilizes an inpainting pipeline to generate a modified image
        based on the given prompt and mask.

        Args:
            prompt (str): The text prompt describing the desired modifications.
            image (Image.Image): The original image to be modified.
            mask_image (Image.Image): The mask image indicating areas to be inpainted.
            height (`int`)
            width (`int`)
            num_inference_steps (`int`)
            guidance_scale (`float`)
            num_images_per_prompt (`int`)
            generator (`torch.Generator`)
            max_sequence_length (`int`)
            strength (`float`)

        Returns:
            List[Image.Image]: The list of inpainted image(s) if successful; None if an error occurred.
        """
        output = self._pipeline_mask_inpainting(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            max_sequence_length=max_sequence_length,
            strength=strength,
        )

        if output is None:
            return None

        return output.images

    def _image_to_image(
        self,
        prompt: str,
        image: Image.Image,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        generator: torch.Generator,
        max_sequence_length: int,
    ) -> List[Image.Image] | None:
        """
        Generate an image from a given prompt + provided reference image.

        Image generation pipeline to create an image based on a provided textual prompt.

        Args:
            prompt (`str`)
            height (`int`)
            width (`int`)
            num_inference_steps (`int`)
            guidance_scale (`float`)
            num_images_per_prompt (`int`)
            generator (`torch.Generator`)
            max_sequence_length (`int`)

        Returns:
            List[Image.Image]: The list of generated image(s) if successful; None if an error occurred.
        """
        output = self._pipeline_img2img(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )

        if output is None:
            return None

        return output.images
