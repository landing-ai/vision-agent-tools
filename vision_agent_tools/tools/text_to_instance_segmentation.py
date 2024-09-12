import torch
from PIL import Image
from pydantic import validate_call
from vision_agent_tools.shared_types import BaseTool, VideoNumpy
from vision_agent_tools.models.florence2_sam2 import Florence2SAM2, Florence2SAM2Config
from vision_agent_tools.models.florence2_sam2 import ImageBboxAndMaskLabel


class TextToInstanceSegmentationTool(BaseTool):
    """
    A tool that processes a video or an image with text prompts for detection and segmentation.
    """

    def __init__(self, model_config: Florence2SAM2Config | None = None):
        self._model_config = model_config
        self._model = Florence2SAM2(self._model_config)

    @validate_call(config={"arbitrary_types_allowed": True})
    @torch.inference_mode()
    def __call__(
        self,
        prompts: list[str],
        image: Image.Image | None = None,
        video: VideoNumpy | None = None,
        chunk_length: int | None = 20,
        iou_threshold: float = 0.8,
    ) -> dict[int, dict[int, ImageBboxAndMaskLabel]]:
        """
        Florence2Sam2 model find objects in an image and track objects in a video.
        Args:
            prompt (list[str]): The list of objects to be found.
            image (Image.Image | None): The image to be analyzed.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            chunk_length (int): The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.
        Returns:
            dict[int, ImageBboxMaskLabel]: a dictionary where the first key is the frame index
            then an annotation ID, then an object with the mask, label and possibly bbox (for images)
            for each annotation ID. For example:
                {
                    0:
                        {
                            0: ImageBboxMaskLabel({"mask": np.ndarray, "label": "car"}),
                            1: ImageBboxMaskLabel({"mask", np.ndarray, "label": "person"})
                        },
                    1: ...
                }
        """
        if image is None and video is None:
            raise ValueError("Either 'image' or 'video' must be provided.")
        if image is not None and video is not None:
            raise ValueError("Only one of 'image' or 'video' can be provided.")

        if image is not None:
            result = self._model(prompts=prompts, image=image)
        elif video is not None:
            assert video.ndim == 4, "Video should have 4 dimensions"
            result = self._model(
                prompts=prompts,
                video=video,
                chunk_length=chunk_length,
                iou_threshold=iou_threshold,
            )
        return result
