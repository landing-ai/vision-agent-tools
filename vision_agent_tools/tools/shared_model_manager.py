import asyncio
from typing import Any, Callable, Dict, Optional

from vision_agent_tools.shared_types import BaseTool, Device


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
        self, model_creation_fn: Callable[[], BaseTool], device: Device = Device.CPU
    ) -> None:
        """
        Adds a model to the pool with a device preference.

        Args:
            model_creation_fn (callable): A function that creates the model.
            device (Device): The preferred device for the model.
        """

        class_name = model_creation_fn.__name__  # Get class name from function
        if class_name in self.models:
            print(f"Model '{class_name}' already exists in the pool.")
        else:
            model = model_creation_fn()
            self.models[class_name] = model
            self.model_locks[class_name] = asyncio.Lock()
            self.devices[class_name] = device

    async def fetch_model(self, class_name: str) -> BaseTool:
        """
        Retrieves a model from the pool for safe execution.

        Args:
            class_name (str): Name of the model class.

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
                        # Update current GPU model (for testing)
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
            self.gpu_semaphore.release()  # Release semaphore if held
