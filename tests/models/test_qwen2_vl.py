from vision_agent_tools.models.qwen2_vl import Qwen2VL
import cv2
import numpy as np
import pytest
from PIL import Image


def load_video_frames(video_path: str) -> np.ndarray:
    # Load the video into frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames, axis=0)
    return frames


@pytest.mark.skip(reason="Qwen2VL model does not fit on the current GPU memory")
def test_successful_qwen2vl_for_video(random_video_generator):
    video_path = "tests/shared_data/videos/test_video_5_frames.mp4"
    video_np = load_video_frames(video_path)
    prompt = "Here are some frames of a video. Describe this video in detail"

    run_inference = Qwen2VL()
    answer = run_inference(video=video_np, prompt=prompt)

    assert len(answer) > 0
    assert len(answer[0]) > 0


@pytest.mark.skip(reason="Qwen2VL model does not fit on the current GPU memory")
def test_successful_qwen2vl_for_images():
    test_image = "car.jpg"

    image = Image.open(f"tests/shared_data/images/{test_image}")

    run_inference = Qwen2VL()

    answer = run_inference(images=[image], prompt="what is the color of the car?")
    assert len(answer) > 0
    assert len(answer[0]) > 0
