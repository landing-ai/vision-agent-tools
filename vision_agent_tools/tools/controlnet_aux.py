import torch
from controlnet_aux import OpenposeDetector
from PIL import Image

from vision_agent_tools.shared_types import Device


class Image2Pose:
    """
    A class that simplifies human pose detection using a pre-trained Openpose model.

    This class provides a convenient way to run pose detection on images using a
    pre-trained Openpose model from the `controlnet_aux` library. It takes a PIL
    Image object as input and returns the predicted pose information.

    Args:
        None
    """

    def __init__(self):
        """
        Initializes the Image2Pose object with a pre-trained Openpose detector.

        This method loads a pre-trained Openpose model from the specified model hub
        ("lllyasviel/Annotators" in this case). The loaded detector is stored as an
        attribute for future use.
        """
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Performs pose detection on a PIL image and returns the results.

        This method takes a PIL Image object as input and runs the loaded Openpose
        detector on it. The predicted pose information is then resized to match the
        original image size and returned.

        Args:
            image (PIL.Image): The input image for pose detection.

        Returns:
            PIL.Image: The image with the predicted pose information (format might vary
                      depending on the specific OpenposeDetector implementation).
        """

        image = image.convert("RGB")
        original_size = image.size
        pose = self.detector(image)
        pose = pose.resize(original_size)
        return pose

    def to(self, device: Device):
        self.detector.to(device.value)
