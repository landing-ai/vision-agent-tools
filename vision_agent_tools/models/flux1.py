import torch
from PIL import Image
from pydantic import BaseModel, Field
from pydantic import ConfigDict, validate_arguments
from vision_agent_tools.shared_types import BaseMLModel, Device
from diffusers import FluxPipeline, FluxInpaintPipeline
from enum import Enum
import logging

_LOGGER = logging.getLogger(__name__)


class Flux1Config(BaseModel):
    hf_model: str = Field(
        default="black-forest-labs/FLUX.1-schnell",
        description="Name of the HuggingFace model",
    )
    task: str = Field(
        default="generation",
        description="Task to perform using the model. Default is 'generation'.",
    )
    height: int = 512
    width: int = 512
    guidance_scale: float = 3.5
    num_inference_steps: int = 10
    max_sequence_length: int = 256
    strength: float = 0.85
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
        guidance_scale: float,
        num_inference_steps: int,
        max_sequence_length: int,
        seed: int = 0,
    ) -> Image.Image | None:
        """
        Generate an image from a given prompt.

        Image generation pipeline to create an image based on a provided textual prompt.

        Args:
            prompt (str): The text prompt for generating the image.

        Returns:
            Optional[Image.Image]: The generated image if successful; None if an error occurred.
        """
        try:
            generator = torch.Generator("cpu").manual_seed(seed)
            image = self._pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=generator,
            ).images[0]
            return image

        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred during image generation: {e}")
            raise RuntimeError(
                "An unexpected error occurred during image generation."
            ) from e

    def _inpaint_image(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        max_sequence_length: int,
        strength: float,
        seed: int = 0,
    ) -> Image.Image | None:
        """
        Inpaint an image using a given prompt and a mask image.

        This method utilizes an inpainting pipeline to generate a modified image
        based on the given prompt and mask.

        Args:
            prompt (str): The text prompt describing the desired modifications.
            image (Image.Image): The original image to be modified.
            mask_image (Image.Image): The mask image indicating areas to be inpainted.

        Returns:
            Optional[Image.Image]: The inpainted image if successful; None if an error occurred.
        """
        try:
            generator = torch.Generator("cpu").manual_seed(seed)
            image = self._inpaint_pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=generator,
                image=image,
                mask_image=mask_image,
                strength=strength,
            ).images[0]
            return image
        except Exception as e:
            _LOGGER.error(
                f"An unexpected error occurred during image mask inpainting: {e}"
            )
            raise RuntimeError(
                "An unexpected error occurred during image mask inpainting."
            ) from e

    def __init__(self, model_config: Flux1Config | None = None):
        """
        Initializes the Flux1 image generation tool.
        Loads the pre-trained Flux1 model from HuggingFace and sets the device to run the model on.
        """
        self.model_config = model_config or Flux1Config()
        dtype = torch.bfloat16

        self._pipeline = FluxPipeline.from_pretrained(
            self.model_config.hf_model, torch_dtype=dtype
        )
        self._pipeline.enable_sequential_cpu_offload()

        self._inpaint_pipeline = FluxInpaintPipeline.from_pretrained(
            self.model_config.hf_model, torch_dtype=dtype
        )
        self._inpaint_pipeline.enable_sequential_cpu_offload()

    @torch.inference_mode()
    @validate_arguments(config=config)
    def __call__(
        self,
        task: Flux1Task,
        prompt: str,
        image: Image.Image | None = None,  # Only used for inpainting
        mask_image: Image.Image | None = None,  # Only used for inpainting
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
        max_sequence_length: int | None = None,
        strength: float | None = None,  # Only used for inpainting
        seed: int = 0,
    ) -> Image.Image | None:
        """
        Performs object detection on an image using the Flux1 model.

        Args:
            prompts str: A prompt to generate an image from.
                        Currently, only one prompt is supported (single string)
                        and the model will generate a single image.

            task (Flux1Task): The task to perform using the model - either image generation ("generation") or mask inpainting ("inpainting").
            prompt (str): A prompt to generate an image from. Currently, only one prompt is supported (single string) and the model will generate a single image.
            image (Image.Image | None, optional): The image to inpaint. Defaults to None. Only used for inpainting.
            mask_image (Image.Image | None, optional): The mask image to use for inpainting. Defaults to None. Only used for inpainting.
            height (int | None, optional): The height of the generated image. Defaults to None.
            width (int | None, optional): The width of the generated image. Defaults to None.
            guidance_scale (float | None, optional): The guidance scale for image generation. Defaults to None.
            num_inference_steps (int | None, optional): The number of inference steps for image generation. Defaults to None.
            max_sequence_length (int | None, optional): The maximum sequence length for image generation. Defaults to None.
            strength (float | None, optional): The strength for inpainting. Defaults to None. Only used for inpainting.
            seed (int, optional): The seed for random number generation. Defaults to 0.

        Returns:
            Image.Image | None: The output image if the Flux1 process is successful;
                None if an error occurred.
        """
        if task == Flux1Task.IMAGE_GENERATION:
            generated_img = self._generate_image(
                prompt=prompt,
                height=height or self.model_config.height,
                width=width or self.model_config.width,
                guidance_scale=guidance_scale or self.model_config.guidance_scale,
                num_inference_steps=num_inference_steps
                or self.model_config.num_inference_steps,
                max_sequence_length=max_sequence_length
                or self.model_config.max_sequence_length,
                seed=seed or 0,
            )
        elif task == Flux1Task.MASK_INPAINTING:
            generated_img = self._inpaint_image(
                image=image,
                mask_image=mask_image,
                prompt=prompt,
                height=height or self.model_config.height,
                width=width or self.model_config.width,
                guidance_scale=guidance_scale or self.model_config.guidance_scale,
                num_inference_steps=num_inference_steps
                or self.model_config.num_inference_steps,
                max_sequence_length=max_sequence_length
                or self.model_config.max_sequence_length,
                strength=strength or self.model_config.strength,
                seed=seed or 0,
            )
        else:
            _LOGGER.error(f"Unsupported task: {task}")
            raise ValueError(
                f"Unsupported task: {task}. Supported tasks are: {', '.join([task.value for task in Flux1Task])}."
            )

        return generated_img

    def to(self, device: Device):
        self._pipeline.to(device=device.value)
        self._device = device
