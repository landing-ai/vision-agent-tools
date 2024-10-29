from PIL import Image

from vision_agent_tools.models.florence2 import Florence2, Florence2Config
from vision_agent_tools.shared_types import PromptTask, Florence2ModelName


def test_large_model_od_image(shared_large_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.OBJECT_DETECTION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    request = {"images": [image], "task": task, "prompt": prompt}
    response = shared_large_model(**request)
    assert response == [
        {
            "bboxes": [],
            "labels": [],
        }
    ]


def test_small_model_od_image(shared_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.OBJECT_DETECTION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    request = {"images": [image], "task": task, "prompt": prompt}
    response = shared_model(**request)
    assert response == [
        {
            "bboxes": [],
            "labels": [],
        }
    ]


def test_od_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    model_zip_path = "tests/models/florence2/data/models/od_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    task = PromptTask.OBJECT_DETECTION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    small_model = Florence2(config)
    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = small_model(**payload)
    assert response == [
        {
            "bboxes": [
                [738.3040161132812, 1373.18408203125, 881.6640625, 1557.5040283203125]
            ],
            "labels": ["screw"],
        }
    ]


def test_large_model_base_with_small_model_od_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    model_zip_path = "tests/models/florence2/data/models/od_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    task = PromptTask.OBJECT_DETECTION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_LARGE,
        fine_tuned_model_path=model_path,
    )
    large_model = Florence2(config)
    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = large_model(**payload)
    assert response == [
        {
            "bboxes": [
                [738.3040161132812, 1373.18408203125, 881.6640625, 1557.5040283203125]
            ],
            "labels": ["screw"],
        }
    ]


def test_od_ft_and_base_and_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    model_zip_path = "tests/models/florence2/data/models/od_checkpoint.zip"
    model_path = unzip_model(model_zip_path)
    task = PromptTask.OBJECT_DETECTION
    # cannot have prompt
    prompt = ""
    image = Image.open(image_path)

    config = Florence2Config(
        model_name=Florence2ModelName.FLORENCE_2_BASE_FT,
        fine_tuned_model_path=model_path,
    )
    small_model = Florence2(config)
    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = small_model(**payload)
    assert response == [
        {
            "bboxes": [
                [738.3040161132812, 1373.18408203125, 881.6640625, 1557.5040283203125]
            ],
            "labels": ["screw"],
        }
    ]

    # running prediction again without fine_tuning should reset the model to its base
    small_model.load_base()
    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = small_model(**payload)
    assert response == [
        {
            "bboxes": [],
            "labels": [],
        }
    ]

    # running prediction again with fine_tuning
    small_model.fine_tune(model_path)
    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = small_model(**payload)
    assert response == [
        {
            "bboxes": [
                [738.3040161132812, 1373.18408203125, 881.6640625, 1557.5040283203125]
            ],
            "labels": ["screw"],
        }
    ]
