import numpy as np
import pytest
from PIL import Image

from vision_agent_tools.models.sam2 import Sam2, Sam2Config
from vision_agent_tools.shared_types import Device, ODResponse


def test_sam2_point_segmentation_image(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)

    input_points = np.array([[110, 120]])
    input_label = np.array([1])
    input_box = None
    multimask_output = False

    response = shared_model(
        images=[test_image],
        input_points=input_points,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
    )

    assert len(response) == 1  # frames
    frame = response[0]
    assert len(frame) == 1  # annotations
    annotations = frame[0]

    assert annotations.keys() == {"id", "score", "mask", "logits"}
    assert annotations["id"] == 0
    assert annotations["score"] == 0.9140625
    reverted_masks = rle_decode_array(annotations["mask"])
    assert reverted_masks.shape == test_image.size[::-1]
    assert annotations["logits"].shape == (256, 256)


def test_sam2_box_segmentation_image(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)

    input_points = None
    input_label = None
    input_box = np.array(
        [
            [60, 240, 380, 420],
            [65, 250, 150, 340],
        ]
    )
    multimask_output = False

    response = shared_model(
        images=[test_image],
        input_points=input_points,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
    )

    assert len(response) == 1  # frames
    frame = response[0]
    assert len(frame) == 2  # annotations
    expected_scores = [0.953125, 0.921875]
    for idx, (score, annotation) in enumerate(zip(expected_scores, frame)):
        assert annotation.keys() == {"id", "score", "mask", "logits"}
        assert annotation["id"] == idx
        assert annotation["score"] == score
        reverted_masks = rle_decode_array(annotation["mask"])
        assert reverted_masks.shape == test_image.size[::-1]
        assert annotation["logits"].shape == (256, 256)


def test_sam2_empty_bounding_box(shared_model):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    test_image = Image.open(image_path)

    empty_bboxes = [ODResponse(labels=[], bboxes=[])]

    response = shared_model(images=[test_image], bboxes=empty_bboxes)

    assert response == [[]]


def test_sam2_invalid_media(shared_model):
    with pytest.raises(ValueError):
        shared_model(images=["invalid media"])  # Invalid type for image

    with pytest.raises(ValueError):
        shared_model(images=[None])  # No image provided


def test_sam2_video_detection_segmentation(shared_model, rle_decode_array):
    image_path = "tests/shared_data/images/tomatoes.jpg"
    # Load a sample image and create a test video
    pil_test_image = Image.open(image_path)
    test_image = np.array(pil_test_image, dtype=np.uint8)
    test_video = np.array([test_image, np.zeros(test_image.shape, dtype=np.uint8)])

    input_points = np.array([[130, 170]])
    input_label = np.array([1])

    response = shared_model(
        video=test_video, input_points=input_points, input_label=input_label
    )

    assert len(response) == 2  # frames
    for frame in response:
        assert len(frame) == 1  # annotations
        annotation = frame[0]
        assert annotation.keys() == {"id", "score", "mask", "logits"}
        assert annotation["id"] == 0
        assert annotation["score"] is None
        reverted_masks = rle_decode_array(annotation["mask"])
        assert reverted_masks.shape == pil_test_image.size[::-1]
        assert annotation["logits"] is None


@pytest.fixture(scope="module")
def shared_model():
    return Sam2(
        model_config=Sam2Config(
            hf_model="facebook/sam2-hiera-large",
            device=Device.GPU,
        )
    )
