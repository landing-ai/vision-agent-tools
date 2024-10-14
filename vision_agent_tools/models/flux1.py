import torch
from PIL import Image
from pydantic import BaseModel, Field
from pydantic import ConfigDict, validate_arguments
from vision_agent_tools.shared_types import BaseMLModel, Device
from diffusers import FluxPipeline, FluxInpaintPipeline
from enum import Enum
from typing import List
import logging
import random

_LOGGER = logging.getLogger(__name__)


class Flux1Config(BaseModel):
    hf_model: str = Field(
        default="black-forest-labs/FLUX.1-schnell",
        description="Name of the HuggingFace model",
    )
    device: Device = Field(
        default=(
            Device.GPU
            if torch.cuda.is_available()
            else Device.MPS
            if torch.backends.mps.is_available()
            else Device.CPU
        ),
        description="Device to run the model on. Options are 'cpu', 'gpu', and 'mps'. Default is the first available GPU.",
    )


class Flux1Task(str, Enum):
    IMAGE_GENERATION = "generation"
    MASK_INPAINTING = "inpainting"


class Flux1(BaseMLModel):
    """
    Tool for object detection using the pre-trained Flux1 model.
    This tool takes a prompt as input and generates an image using the Flux1 model.
    """

    config = ConfigDict(arbitrary_types_allowed=True)

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
            Optional[Image.Image]: The generated image if successful; None if an error occurred.
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

        return output

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
            Optional[Image.Image]: The inpainted image if successful; None if an error occurred.
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

        return output

    def __init__(
        self,
        model_config: Flux1Config | None = None,
    ):
        """
        Initializes the Flux1 image generation tool.
        Loads the pre-trained Flux1 model from HuggingFace and enables sequential CPU offload.

        Args:
            - task (Flux1Task): The task to perform using the model:
                either image generation ("generation")
                or mask inpainting ("inpainting").
            - model_config: The configuration for the model, hf_model, and device.
        """
        self.model_config = model_config or Flux1Config()
        dtype = torch.bfloat16
        self._pipeline = None

        self._pipeline_img_generation = FluxPipeline.from_pretrained(
            self.model_config.hf_model, torch_dtype=dtype
        )
        self._pipeline_img_generation.enable_sequential_cpu_offload()

        self._pipeline_mask_inpainting = FluxInpaintPipeline.from_pretrained(
            self.model_config.hf_model, torch_dtype=dtype
        )
        self._pipeline_mask_inpainting.enable_sequential_cpu_offload()

    @torch.inference_mode()
    @validate_arguments(config=config)
    def __call__(
        self,
        prompt: str,
        task: Flux1Task = Flux1Task.IMAGE_GENERATION,
        image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
        height: int = 1024,
        width: int = 1024,
        strength: float | None = 0.6,
        num_inference_steps: int | None = 28,
        guidance_scale: float | None = 3.5,
        num_images_per_prompt: int | None = 1,
        max_sequence_length: int | None = 512,
        seed: int | None = None,
    ) -> List[Image.Image] | Image.Image | None:
        """
        Performs object detection on an image using the Flux1 model.

        Args:
            - prompt (str): The text prompt describing the desired modifications.
            - image (Image.Image): The original image to be modified.
            - mask_image (Image.Image): The mask image indicating areas to be inpainted.
            - height (`int`, *optional*):
                The height in pixels of the generated image.
                This is set to 1024 by default for the best results.
            - width (`int`, *optional*):
                The width in pixels of the generated image.
                This is set to 1024 by default for the best results.
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
            - strength (`float`, *optional*, defaults to 0.6):
                Indicates extent to transform the reference `image`.
                Must be between 0 and 1.
                A value of 1 essentially ignores `image`.
            - seed (`int`, *optional*): The seed to use for the random number generator.
                If not provided, a random seed is used.

        Returns:
            Image.Image | None: The output image if the Flux1 process is successful;
                None if an error occurred.
        """

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}"
            )

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator("cpu").manual_seed(seed)
        output = None

        if task == Flux1Task.IMAGE_GENERATION:
            output = self._generate_image(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                max_sequence_length=max_sequence_length,
            )
        elif task == Flux1Task.MASK_INPAINTING:
            if image.size != mask_image.size:
                raise ValueError("The image and mask image should have the same size.")

            if height is None:
                height = image.height
            if width is None:
                width = image.width

            output = self._inpaint_image(
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
        else:
            raise ValueError(
                f"Unsupported task: {self.task}. Supported tasks are: {', '.join([task.value for task in Flux1Task])}."
            )

        if num_images_per_prompt == 1:
            return output.images[0]

        return output.images

    def to(self, device: Device):
        self._pipeline.to(device=device.value)
        self._device = device
