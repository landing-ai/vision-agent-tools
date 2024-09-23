# Shared Model Manager

The `SharedModelManager` class is designed to manage and facilitate the use of machine learning models across different devices, such as CPUs and GPUs, within an asynchronous environment.
It ensures safe and efficient execution of these models, particularly in scenarios where GPU resources need to be shared exclusively among multiple models.
The manager coordinates access to the shared GPU, preventing conflicts when multiple models require it.
Models are only loaded into memory when needed using the `fetch_model` function.

- `add()`: Registers a machine learning model class with the manager. The actual model instance is not loaded at this point.
- `fetch_model()`: Retrieves the previously added model class and creates (loads) the actual model instance. This function utilizes PyTorch interface `to`, to handle device (CPU/GPU) allocation based on availability.


The usage example demonstrates adding models and then using them with their respective functionalities.

⚠️ ❕: We should ALWAYS **add model instance on CPU** to the pool. This avoids overwhelming the GPU memory, and model pool will automatically put it in GPU when the model is fetched..


```python
model_pool = SharedModelManager()

# Add models instance to the pool
model_pool.add(QRReader())
model_pool.add(Owlv2(model_config=OWLV2Config(device=Device.CPU)))

# Read image
image = Image.open("path/to/your/image.jpg")

# Use QRReader model
async def use_qr_reader():
    # Read image
    image = Image.open("path/to/your/image.jpg")

    qr_reader = await model_pool.fetch_model(QRReader.__name__)
    detections = qr_reader(image)
    # Process detections ...

# Use Owlv2 model
async def use_owlv2():
    # Read image
    image = Image.open("path/to/your/image.jpg")

    owlv2 = await model_pool.fetch_model(Owlv2.__name__)
    prompts = ["a photo of a cat", "a photo of a dog"]
    results = owlv2(image, prompts=prompts)
    # Process results ...

```

::: vision_agent_tools.tools.shared_model_manager
