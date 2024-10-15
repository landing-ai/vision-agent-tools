import os
import shutil
import logging
import tempfile

import cv2
import pytest
import numpy as np

from vision_agent_tools.shared_types import Florence2ModelName
from vision_agent_tools.models.florence2_ft import Florence2Ft

logging.basicConfig(level=logging.INFO)


# @pytest.fixture(scope="session")
# def large_model():
#     return Florence2Ft(Florence2ModelName.FLORENCE_2_LARGE)


@pytest.fixture(scope="session")
def small_model():
    return Florence2Ft(Florence2ModelName.FLORENCE_2_BASE_FT)


@pytest.fixture
def unzip_model(tmp_path):
    def handler(model_zip_path):
        local_model_path = _unzip(model_zip_path, tmp_path, folder_name="model")
        return os.path.join(local_model_path, "checkpoint")

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


def _unzip(zip_file_path: str, tmp_path: str, folder_name) -> None:
    local_file_path = os.path.join(tmp_path, folder_name)
    os.makedirs(local_file_path, exist_ok=True)
    shutil.unpack_archive(zip_file_path, local_file_path, "zip")
    return local_file_path
