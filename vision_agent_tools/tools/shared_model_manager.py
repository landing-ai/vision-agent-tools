import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from vision_agent_tools.shared_types import BaseTool, Device

_LOGGER = logging.getLogger(__name__)

class SharedModelManager:
    def __init__(self) -> None:
        self.models: Dict[str, BaseTool] = {}  # store models with class name as key
        self.model_locks: Dict[str, asyncio.Lock] = {}  # store locks for each model
        self.devices: Dict[str, Device] = {}  # store device preference for each model
        self.current_gpu_model: Optional[str] = (
            None  # Track the model currently using GPU
        )

        # Semaphore for exclusive GPU access
        self.gpu_semaphore = asyncio.Semaphore(1)

    def add(
        self, model: BaseTool, device: Device = Device.CPU,
    ) -> str:
        """
        Adds a model to the pool with a device preference.

        Args:
            model (Basetool): The modal instance to be added to the pool, it should implement the BaseTool interface.
            device (Device): The preferred device for the model.

        Returns:
            str: The model ID to be used for fetching the model.
        """

        class_name = model.__class__.__name__  # Tool name
        model_name = model.model    # Model name used by the tool
        model_id = f"{class_name}.{model_name}"

        if model_id in self.models:
            _LOGGER.warning(f"Model '{model_id}' already exists in the pool.")
        else:
            self.models[model_id] = model
            self.model_locks[model_id] = asyncio.Lock()
            self.devices[model_id] = device

        return model_id

    async def fetch_model(self, model_id: str) -> BaseTool:
        """
        Retrieves a model from the pool for safe execution.

        Args:
            model_id (str): Id to access the model in the pool.

        Returns:
            Any: The retrieved model instance.
        """

        if class_name not in self.models:
            raise ValueError(f"Model '{class_name}' not found in the pool.")

        model = self.models[class_name]
        lock = self.model_locks[class_name]
        device = self.devices[class_name]

        async def get_model_with_lock() -> Any:
            async with lock:
                if device == Device.GPU:
                    # Acquire semaphore if needed
                    async with self.gpu_semaphore:
                        # Move existing GPU model to CPU
                        exisitng = self._get_current_gpu_model()
                        if exisitng:
                            await self._move_to_cpu(exisitng)

                        # Update current GPU model
                        self.current_gpu_model = class_name
                        model.to(Device.GPU)
                return model

        return await get_model_with_lock()

    def _get_current_gpu_model(self) -> Optional[str]:
        """
        Returns the class name of the model currently using the GPU (if any).
        """
        return self.current_gpu_model

    async def _move_to_cpu(self, class_name: str) -> None:
        """
        Moves a model to CPU and releases the GPU semaphore (if held).
        """
        model = self.models[class_name]
        model.to(Device.CPU)
        if self.current_gpu_model == class_name:
            self.current_gpu_model = None
