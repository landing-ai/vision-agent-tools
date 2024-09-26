import cv2
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
    prompts = ["a car", "a tree"]

    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    owlv2 = Owlv2(model_config=OWLV2Config(max_batch_size=2))
    results = owlv2(prompts=prompts, video=frames)
    assert len(results) > 0


def test_successful_image_detection_with_nms():
    test_image = "surfers_with_shark.png"
    prompts = ["surfer", "shark"]

    image = Image.open(f"tests/tools/data/owlv2/{test_image}")

    owlv2 = Owlv2(model_config=OWLV2Config(confidence=0.2, nms_threshold=0.3))
    results = owlv2(prompts=prompts, image=image)

    # without NMS (nms_threshold=1), there will be 4 detections
    # shark 0.66888028383255 [118.4, 166.0, 281.6, 238.59]
    # surfer 0.2894721031188965 [339.12, 129.7, 386.24, 217.09] <--- removed by NMS
    # surfer 0.5477967262268066 [340.22, 142.03, 388.97, 199.28]
    # surfer 0.4184592366218567 [165.45, 282.42, 203.8, 359.12]
    assert len(results[0]) == 3

    for pred in results[0]:
        assert pred.label in prompts
