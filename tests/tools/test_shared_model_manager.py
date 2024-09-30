from unittest.mock import AsyncMock, MagicMock

import pytest

from vision_agent_tools.shared_types import BaseTool, Device
from vision_agent_tools.tools.shared_model_manager import SharedModelManager


@pytest.fixture
def model_pool():
    return SharedModelManager()


class MockBaseModel(AsyncMock, BaseTool):
    pass


@pytest.mark.asyncio
async def test_add_model(model_pool: SharedModelManager):
    test_model = MockBaseModel()
    model_id = model_pool.add(test_model)
    assert len(model_pool.models) == 1
    assert model_id in model_pool.models

    model_id2 = model_pool.add(test_model)  # Duplicate addition
    assert len(model_pool.models) == 1
    assert model_id2 in model_pool.models
    assert model_id2 == model_id

    model_id3 = model_pool.add(MockBaseModel())  # Not duplicate addition
    assert len(model_pool.models) == 2
    assert model_id3 in model_pool.models
    assert model_id3 != model_id


@pytest.mark.asyncio
async def test_get_model_gpu(model_pool: SharedModelManager):
    model = MockBaseModel()
    model.to = MagicMock()

    model_id = model_pool.add(model)

    model_to_get = model_pool.fetch_model(model_id)

    assert model_to_get.to.call_count == 1
    model_to_get.to.assert_called_once_with(Device.GPU)  # Verify to was called with GPU


@pytest.mark.asyncio
async def test_get_model_not_found(model_pool):
    with pytest.raises(ValueError):
        model_pool.fetch_model("NonexistentModel")


@pytest.mark.asyncio
async def test_swap_model_in_gpu(model_pool: SharedModelManager):
    model_a = MockBaseModel()
    model_a.to = MagicMock()

    model_b = MockBaseModel()
    model_b.to = MagicMock()

    model_a_id = model_pool.add(model_a)
    model_b_id = model_pool.add(model_b)

    # Get Model1 first, should use GPU
    model1_to_get = model_pool.fetch_model(model_a_id)
    assert model1_to_get is not None
    assert model_pool._get_current_gpu_model() == model_a_id

    # Get Model2, should move Model1 to CPU and use GPU
    model2_to_get = model_pool.fetch_model(model_b_id)
    assert model2_to_get is not None
    assert model_pool._get_current_gpu_model() == model_b_id
