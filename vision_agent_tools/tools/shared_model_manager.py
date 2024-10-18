import asyncio
import logging
from typing import Dict, Optional

from vision_agent_tools.shared_types import BaseTool, Device

_LOGGER = logging.getLogger(__name__)


class SharedModelManager:
    def __init__(self) -> None:
        self.models: Dict[str, BaseTool] = {}  # store models with class name as key
        self.model_locks: Dict[str, asyncio.Lock] = {}  # store locks for each model
        self.current_gpu_model: Optional[str] = (
            None  # Track the model currently using GPU
        )

        # Semaphore for exclusive GPU access
        self.gpu_semaphore = asyncio.Semaphore(1)

    def add(self, model: BaseTool) -> str:
        """
        Adds a model to the pool with a device preference.

        Args:
            model (Basetool): The modal instance to be added to the pool, it should implement the BaseTool interface.
            device (Device): The preferred device for the model.

        Returns:
            str: The model ID to be used for fetching the model.
        """

        class_name = model.__class__.__name__  # Tool name
        model_name = model.model  # Model name used by the tool
        model_id = f"{class_name}.{model_name}"

        if model_id in self.models:
            _LOGGER.warning(f"Model '{model_id}' already exists in the pool.")
        else:
            self.models[model_id] = model
            self.model_locks[model_id] = asyncio.Lock()

        return model_id

    def fetch_model(self, model_id: str) -> BaseTool:
        """
        Retrieves a model from the pool for safe execution.

        Args:
            model_id (str): Id to access the model in the pool.

        Returns:
            Any: The retrieved model instance.
        """

        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in the pool.")

        model = self.models[model_id]

        # Move existing GPU model to CPU
        existing = self._get_current_gpu_model()
        if existing:
            model = self.models[model_id]
            model.to(Device.CPU)
            if self.current_gpu_model == model_id:
                self.current_gpu_model = None

        # Update current GPU model
        self.current_gpu_model = model_id
        model.to(Device.GPU)
        return model

    def _get_current_gpu_model(self) -> Optional[str]:
        """
        Returns the class name of the model currently using the GPU (if any).
        """
        return self.current_gpu_model

    def to(self, device: Device):
        self.model.to(device)
        return self
