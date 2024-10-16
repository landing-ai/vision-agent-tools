import pytest

from vision_agent_tools.models.florence2 import Florence2
from vision_agent_tools.models.model_registry import get_model_class


def test_get_model_class_valid_model():
    valid_model_name = "florence2"
    model_class = get_model_class(valid_model_name)
    model_instance = model_class()
    assert isinstance(model_instance(), Florence2)


def test_get_model_class_invalid_model():
    invalid_model_name = "nonexistent_model"
    with pytest.raises(
        ValueError,
        match=f"Model '{invalid_model_name}' is not registered in the model registry.",
    ):
        get_model_class(invalid_model_name)
