import pytest
from PIL import Image
from pydantic import ValidationError

from vision_agent_tools.shared_types import PromptTask


def test_no_images_and_no_video(shared_model):
    task = PromptTask.OBJECT_DETECTION

    payload = {"task": task}
    with pytest.raises(ValidationError) as exc:
        shared_model(**payload)
        assert exc.value == "video or images is required"


def test_images_and_video(shared_model, bytes_to_np):
    image_path = "tests/shared_data/images/cereal.jpg"
    video_path = "tests/shared_data/videos/shark_10fps.mp4"
    task = PromptTask.OBJECT_DETECTION
    image = Image.open(image_path)

    with open(video_path, "rb") as f:
        video_bytes = f.read()
        video = bytes_to_np(video_bytes)

    payload = {"images": [image], "video": video, "task": task}
    with pytest.raises(ValidationError) as exc:
        shared_model(**payload)
        assert exc.value == "Only one of them are required: video or images"


def test_batch_size_validation(shared_model):
    task = PromptTask.OCR
    image = Image.new("RGB", (224, 224))

    invalid_batch_sizes = [0, -1]
    for batch_size in invalid_batch_sizes:
        with pytest.raises(ValidationError):
            shared_model(task=task, images=[image], batch_size=batch_size)
