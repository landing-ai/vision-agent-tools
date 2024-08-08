import cv2
import numpy as np
from PIL import Image
from vision_agent_tools.tools.internlm_xcomposer2 import InternLMXComposer2


def video_to_numpy_array(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frames = []

    # Read until the end of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Append the frame to the list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    # Convert list of frames to a NumPy array
    video_array = np.array(frames)

    return video_array


def test_successful_internlm_xcomposer2_for_video():
    test_video = "london_telephone_box.mp4"

    video_np = video_to_numpy_array(
        f"tests/tools/data/internlm_xcomposer2/{test_video}"
    )
    prompt = "Here are some frames of a video. Describe this video in detail"

    run_inference = InternLMXComposer2()

    answer = run_inference(video=video_np, prompt=prompt)

    assert len(answer) > 0


def test_successful_internlm_xcomposer2_for_images():
    test_image = "car.jpg"

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    run_inference = InternLMXComposer2()

    answer = run_inference(image=image, prompt="what is the color of the car?")

    assert len(answer) > 0
