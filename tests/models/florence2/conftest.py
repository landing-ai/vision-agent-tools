import pytest

from vision_agent_tools.shared_types import Florence2ModelName
from vision_agent_tools.models.florence2 import Florence2, Florence2Config


@pytest.fixture(scope="package")
def shared_large_model():
    # do not fine-tune this model, if you do, remember to reset it to the base model
    # by calling shared_model.load_base()
    config = Florence2Config(model_name=Florence2ModelName.FLORENCE_2_LARGE)
    return Florence2(config)


@pytest.fixture(scope="package")
def shared_model():
    # do not fine-tune this model, if you do, remember to reset it to the base model
    # by calling shared_model.load_base()
    config = Florence2Config(model_name=Florence2ModelName.FLORENCE_2_BASE_FT)
    return Florence2(config)
