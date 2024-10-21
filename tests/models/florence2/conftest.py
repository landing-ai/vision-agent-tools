import os
import shutil

import pytest

from vision_agent_tools.models.florence2 import Florence2
from vision_agent_tools.shared_types import Florence2ModelName


@pytest.fixture
def large_model():
    return Florence2(Florence2ModelName.FLORENCE_2_LARGE)


@pytest.fixture
def small_model():
    return Florence2(Florence2ModelName.FLORENCE_2_BASE_FT)


@pytest.fixture(scope="session")
def shared_large_model():
    # do not fine-tune this model, if you do, remember to reset it to the base model
    # by calling shared_model.load_base()
    return Florence2(Florence2ModelName.FLORENCE_2_LARGE)


@pytest.fixture(scope="session")
def shared_model():
    # do not fine-tune this model, if you do, remember to reset it to the base model
    # by calling shared_model.load_base()
    return Florence2(Florence2ModelName.FLORENCE_2_BASE_FT)


@pytest.fixture
def unzip_model(tmp_path):
    def handler(model_zip_path):
        local_model_path = _unzip(model_zip_path, tmp_path, folder_name="model")
        return os.path.join(local_model_path, "checkpoint")

    return handler


def _unzip(zip_file_path: str, tmp_path: str, folder_name) -> None:
    local_file_path = os.path.join(tmp_path, folder_name)
    os.makedirs(local_file_path, exist_ok=True)
    shutil.unpack_archive(zip_file_path, local_file_path, "zip")
    return local_file_path
