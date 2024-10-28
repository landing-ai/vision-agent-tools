import json
from PIL import Image

from vision_agent_tools.models.florence2 import Florence2, Florence2Config
from vision_agent_tools.shared_types import PromptTask, Florence2ModelName


def test_caption_to_phrase_grounding_cereal(shared_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
    prompt = "screw"
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = shared_model(**payload)
    # without removing the whole container box, the model will detect one bbox
    # in this image
    assert response == [
        {
            "bboxes": [],
            "labels": [],
        }
    ]


def test_caption_to_phrase_grounding_sheep(shared_large_model):
    image_path = "tests/shared_data/images/sheep_aerial.jpg"
    task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
    prompt = "sheep"
    image = Image.open(image_path)

    payload = {
        "images": [image],
        "task": task,
        "prompt": prompt,
    }
    response = shared_large_model(**payload)

    assert len(response) == 1
    # this test case test the nms postprocessing where the model would output
    # more than 14 bboxes for this image without nms
    assert len(response[0]["bboxes"]) == 14


def test_caption_to_phrase_grounding_car_with_nms(shared_model):
    image_path = "tests/shared_data/images/car.jpg"
    task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
    prompt = "car"
    image = Image.open(image_path)

    payload = {"images": [image], "task": task, "prompt": prompt, "nms_threshold": 0.1}
    response = shared_model(**payload)
    assert response == [
        {
            "labels": ["car"],
            "bboxes": [
                [
                    34.880001068115234,
                    158.16000366210938,
                    582.0800170898438,
                    373.1999816894531,
                ]
            ],
        }
    ]


def test_caption_to_phrase_grounding_video(shared_model, bytes_to_np):
    video_path = "tests/shared_data/videos/shark_10fps.mp4"
    task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
    prompt = "shark"
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        video = bytes_to_np(video_bytes)

    payload = {
        "video": video,
        "task": task,
        "prompt": prompt,
    }
    response = shared_model(**payload)

    assert len(response) == 80
    with open(
        "tests/models/florence2/data/results/caption_to_phrase_grounding_video_results.json",
        "r",
    ) as source:
        expected_response = json.load(source)
    assert response == expected_response


def test_caption_to_phrase_grounding_image_ft(unzip_model):
    image_path = "tests/shared_data/images/cereal.jpg"
    model_zip_path = (
        "tests/shared_data/models/caption_to_phrase_grounding_checkpoint.zip"
    )
    model_path = unzip_model(model_zip_path)
    task = PromptTask.CAPTION_TO_PHRASE_GROUNDING
    prompt = "screw"
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
                [
                    723.968017578125,
                    1373.18408203125,
                    902.14404296875,
                    1577.984130859375,
                ]
            ],
            "labels": ["screw"],
        }
    ]
