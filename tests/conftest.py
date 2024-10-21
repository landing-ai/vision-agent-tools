import logging
import tempfile
from typing import Any

import cv2
import pytest
import numpy as np

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def random_video_generator():
    def _generate_test_video(n_frames=3):
        """
        Generate a test video with n_frames.
        """
        return np.random.randint(0, 255, (n_frames, 300, 300, 3), dtype=np.uint8)

    return _generate_test_video


@pytest.fixture
def rle_decode_array():
    def handler(rle: dict[str, Any]) -> np.ndarray:
        size = rle["size"]
        counts = rle["counts"]

        total_elements = size[0] * size[1]
        flattened_mask = np.zeros(total_elements, dtype=np.uint8)

        current_pos = 0
        for i, count in enumerate(counts):
            if i % 2 == 1:
                flattened_mask[current_pos : current_pos + count] = 1
            current_pos += count

        binary_mask = flattened_mask.reshape(size, order="F")
        return binary_mask
    return handler


@pytest.fixture
def bytes_to_np():
    def handler(video_bytes: bytes):
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(video_bytes)
            fp.flush()
            video_temp_file = fp.name
            cap = cv2.VideoCapture(video_temp_file)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            return np.array(frames)

    return handler
