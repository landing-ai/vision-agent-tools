import pytest
from unittest.mock import AsyncMock, MagicMock

from vision_agent_tools.tools.shared_model_manager import SharedModelManager
from vision_agent_tools.tools.shared_types import BaseTool, Device


@pytest.fixture
def model_pool():
    return SharedModelManager()


class MockBaseModel(AsyncMock, BaseTool):
    pass


@pytest.mark.asyncio
async def test_add_model(model_pool):
    def model_creation_fn():
        return MockBaseModel()

    model_pool.add(model_creation_fn)

    assert len(model_pool.models) == 1
    assert model_creation_fn.__name__ in model_pool.models

    model_pool.add(model_creation_fn)  # Duplicate addition

    assert len(model_pool.models) == 1
    assert model_creation_fn.__name__ in model_pool.models


@pytest.mark.asyncio
async def test_get_model_cpu(model_pool):
    def model_creation_fn():
        model = MockBaseModel()
        model.to = MagicMock()
        return model

    model_pool.add(model_creation_fn)

    model_to_get = await model_pool.get_model(model_creation_fn.__name__)

    assert model_to_get is not None
    assert model_to_get.to.call_count == 0  # No device change for CPU


@pytest.mark.asyncio
async def test_get_model_gpu(model_pool):
    def model_creation_fn():
        model = MockBaseModel()
        model.to = MagicMock()
        model.device = Device.GPU
        return model

    model_pool.add(model_creation_fn)

    model_to_get = await model_pool.get_model(model_creation_fn.__name__)

    assert model_to_get.to.call_count == 1
    model_to_get.to.assert_called_once_with(Device.GPU)  # Verify to was called with GPU


@pytest.mark.asyncio
async def test_get_model_not_found(model_pool):
    with pytest.raises(ValueError):
        await model_pool.get_model("NonexistentModel")


@pytest.mark.asyncio
async def test_get_model_multiple_gpu(model_pool):
    def model_creation_fn_a():
        model = MockBaseModel()
        model.to = MagicMock()
        model.device = Device.GPU  # Set device during creation
        return model

    def model_creation_fn_b():
        model = MockBaseModel()
        model.to = MagicMock()
        model.device = Device.GPU  # Set device during creation
        return model

    model_pool.add(model_creation_fn_a)
    model_pool.add(model_creation_fn_b)

    # Get Model1 first, should use GPU
    model1_to_get = await model_pool.get_model(model_creation_fn_a.__name__)
    assert model1_to_get is not None
    # assert model1_to_get is model1
    assert (
        model_pool._get_current_gpu_model() == model_creation_fn_a.__name__
    )  # Assert device on retrieved model

    # Get Model2, should move Model1 to CPU and use GPU
    model2_to_get = await model_pool.get_model(model_creation_fn_b.__name__)
    assert model2_to_get is not None
    # assert model2_to_get is model2
    assert (
        model_pool._get_current_gpu_model() == model_creation_fn_b.__name__
    )  # Assert device change on Model1
