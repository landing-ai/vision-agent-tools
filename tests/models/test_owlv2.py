import io

from decord import VideoReader, cpu
from PIL import Image

from vision_agent_tools.models.owlv2 import Owlv2, OWLV2Config


def test_successful_image_detection():
    test_image = "000000039769.jpg"
    prompts = ["a photo of a cat", "a photo of a dog"]

    image = Image.open(f"tests/tools/data/owlv2/{test_image}")

    owlv2 = Owlv2()

    results = owlv2(prompts=prompts, image=image)

    assert len(results[0]) > 0

    for pred in results[0]:
        assert pred.label == "a photo of a cat"


def test_successful_video_detection():
    test_video = "test_video_5_frames.mp4"
    file_path = f"tests/tools/data/owlv2/{test_video}"

    with open(file_path, "rb") as f:
        video_bytes = f.read()
    prompts = ["a car", "a tree"]

    video = io.BytesIO(video_bytes)
    video_reader = VideoReader(video, ctx=cpu(0))
    frames = video_reader.get_batch(range(len(video_reader))).asnumpy()

    owlv2 = Owlv2()

    results = owlv2(prompts=prompts, video=frames)

    assert len(results) > 0


def test_successful_image_detection_with_nms():
    test_image = "surfers_with_shark.png"
    prompts = ["a photo of a shark", "a photo of a surfer"]

    image = Image.open(f"tests/tools/data/owlv2/{test_image}")

    owlv2 = Owlv2()
    owlv2_config = OWLV2Config(confidence=0.3, nms_threshold=0.4)
    results = owlv2(prompts=prompts, image=image, model_config=owlv2_config)

    assert len(results[0]) == 3

    for pred in results[0]:
        assert pred.label in ["a photo of a shark", "a photo of a surfer"]