import asyncio
from vision_agent_tools.tools.shared_types import Device


class SharedModelManager:
    def __init__(self):
        self.models = {}  # store models with class name as key
        self.model_locks = {}  # store locks for each model
        self.device = Device.CPU
        self.current_gpu_model = None  # Track the model currently using GPU

        # Semaphore for exclusive GPU access
        self.gpu_semaphore = asyncio.Semaphore(1)

    def add(self, model_creation_fn):
        """
        Adds a model to the pool with a device preference.

        Args:
            model_creation_fn (callable): A function that creates the model.
        """

        class_name = model_creation_fn.__name__  # Get class name from function
        if class_name in self.models:
            print(f"Model '{class_name}' already exists in the pool.")
        else:
            model = model_creation_fn()
            self.models[class_name] = model
            self.model_locks[class_name] = asyncio.Lock()

    async def get_model(self, class_name):
        """
        Retrieves a model from the pool for safe execution.

        Args:
            class_name (str): Name of the model class.

        Returns:
            BaseTool: The retrieved model instance.
        """

        if class_name not in self.models:
            raise ValueError(f"Model '{class_name}' not found in the pool.")

        model = self.models[class_name]
        lock = self.model_locks[class_name]

        async def get_model_with_lock():
            async with lock:
                if model.device == Device.GPU:
                    # Acquire semaphore if needed
                    async with self.gpu_semaphore:
                        # Update current GPU model (for testing)
                        self.current_gpu_model = class_name
                        model.to(Device.GPU)
                return model

        return await get_model_with_lock()

    def _get_current_gpu_model(self):
        """
        Returns the class name of the model currently using the GPU (if any).
        """
        return self.current_gpu_model

    async def _move_to_cpu(self, class_name):
        """
        Moves a model to CPU and releases the GPU semaphore (if held).
        """
        model = self.models[class_name]
        model.to(Device.CPU)
        if self.current_gpu_model == class_name:
            self.current_gpu_model = None
            self.gpu_semaphore.release()  # Release semaphore if held

    def __call__(self, class_name, arguments):
        """
        Decorator for safe and efficient model execution.

        Args:
            class_name (str): Name of the model class.
            arguments (tuple): Arguments for the model call.

        Returns:
            callable: A wrapper function that retrieves the model and executes it.
        """

        async def wrapper():
            model = await self.get_model(class_name)
            return model(arguments)

        return wrapper
