# Shared Model Manager

The `SharedModelManager` class is designed to manage and facilitate the use of machine learning models across different devices, such as CPUs and GPUs, within an asynchronous environment.
It ensures safe and efficient execution of these models, particularly in scenarios where GPU resources need to be shared exclusively among multiple models.

- The `add` function takes the model class as input and stores it with a placeholder.
- The `get_model` function retrieves the model class and lazily loads the actual model instance.


The usage example demonstrates adding models and then using them with their respective functionalities.


```python
# Add models to the pool
model_pool.add(QRReader)
model_pool.add(Owlv2)

# Read image
image = Image.open("path/to/your/image.jpg")

# Use QRReader model
async def use_qr_reader():
    # Read image
    image = Image.open("path/to/your/image.jpg")

    qr_reader = await model_pool.get_model(QRReader.__name__)
    detections = qr_reader(image)
    # Process detections ...

# Use Owlv2 model
async def use_owlv2():
    # Read image
    image = Image.open("path/to/your/image.jpg")

    owlv2 = await model_pool.get_model(Owlv2.__name__)
    prompts = ["a photo of a cat", "a photo of a dog"]
    results = owlv2(image, prompts=prompts)
    # Process results ...

```

::: vision_agent_tools.tools.shared_model_manager
