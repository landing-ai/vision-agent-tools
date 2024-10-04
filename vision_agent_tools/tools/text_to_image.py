from enum import Enum
from PIL import Image
from pydantic import ConfigDict, validate_arguments
from vision_agent_tools.models.flux1 import Flux1Task, Flux1Config
from vision_agent_tools.models.model_registry import get_model_class
from vision_agent_tools.shared_types import BaseTool, Device


class TextToImageModel(str, Enum):
    FLUX1 = "flux1"


class TextToImage(BaseTool):
    """
    Tool to perform image generation or image mask inpainting using text prompts and the specified ML model
    """

    config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: TextToImageModel,
        model_config: Flux1Config | None = None,
    ):
        if model not in TextToImageModel._value2member_map_:
            raise ValueError(
                f"Model '{model}' is not a valid model for {self.__class__.__name__}."
            )

        self._model_name: TextToImageModel = model

        if model == TextToImageModel.FLUX1:
            self._model_config = model_config or Flux1Config()
            self._model_class = get_model_class(model_name=self._model_name)
            model_instance = self._model_class()
            super().__init__(model=model_instance(self._model_config))
        else:
            raise ValueError(f"Model is not supported: '{model}'")

    @validate_arguments(config=config)
    def __call__(
        self,
        task: Flux1Task,
        prompt: str,
        image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
    ) -> Image.Image | None:
        """
        Perform the specified task (Image Generation or Mask Inpainting) using the model and return the results.

        Args:
            task (Flux1Task): The task to perform using the model.
            prompt (str): The text prompt for image generation.
            image (Image.Image): The input image for mask inpainting.
            mask_image (Image.Image): The mask image for mask inpainting.

        Returns:
            Optional[Image.Image]: The generated / inpainted image if successful; None if an error occurred.
        """
        output_image = None

        if self._model_name == TextToImageModel.FLUX1:
            if task == Flux1Task.IMAGE_GENERATION:
                output_image = self.model(task=task, prompt=prompt)
            elif task == Flux1Task.MASK_INPAINTING:
                if image is None or mask_image is None:
                    raise ValueError(
                        "Image and mask_image are required for mask inpainting."
                    )
                output_image = self.model(
                    task=task, prompt=prompt, image=image, mask_image=mask_image
                )
            else:
                raise ValueError(
                    f"Unsupported task: {task}. Supported tasks are: {', '.join([task.value for task in Flux1Task])}."
                )

        return output_image

    def to(self, device: Device):
        self.model.to(device)
        return self
