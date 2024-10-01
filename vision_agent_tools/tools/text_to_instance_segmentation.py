import torch
from PIL import Image
from pydantic import validate_call
from vision_agent_tools.shared_types import BaseTool, VideoNumpy
from vision_agent_tools.models.florence2_sam2 import Florence2SAM2, Florence2SAM2Config
from vision_agent_tools.models.florence2_sam2 import BboxAndMaskLabel


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
        iou_threshold: float = 0.6,
        nms_threshold: float = 1.0,
    ) -> list[list[BboxAndMaskLabel]]:
        """
        Florence2Sam2 model find objects in an image and track objects in a video.
        Args:
            prompt (list[str]): The list of objects to be found.
            image (Image.Image | None): The image to be analyzed.
            video (VideoNumpy | None): A numpy array containing the different images, representing the video.
            chunk_length (int): The number of frames for each chunk of video to analyze. The last chunk may have fewer frames.
            iou_threshold (float): The IoU threshold value used to compare last_predictions and new_predictions objects.
            nms_threshold (float): The non-maximum suppression threshold value used to filter the Florencev2 predictions.

        Returns:
            list[list[ImageBboxMaskLabel]]: a list where the first list contains the frames predictions,
            then the second list contains the annotation, where the annotations are objects with the mask,
            label and bbox (for images) for each annotation. For example:
                [
                    [
                        BboxAndMaskLabel({"mask": np.ndarray, "label": "car", score: 0.9}),
                        BboxAndMaskLabel({"mask", np.ndarray, "label": "person", score: 0.8}),
                    ],
                    ...
                ]
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
                nms_threshold=nms_threshold,
            )
        return result
