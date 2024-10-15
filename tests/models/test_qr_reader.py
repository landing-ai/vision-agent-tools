import pytest
from PIL import Image

from vision_agent_tools.models.qr_reader import QRReader


@pytest.mark.parametrize("expected_text", ["This is a tes"])
def test_successful_qr_detection(expected_text):
    """
    This test verifies that the QRReader successfully detects a known QR code in an image.
    """

    # Load the test images
    for test_image in ["001.jpeg", "002.jpeg"]:
        image = Image.open(f"tests/shared_data/images/{test_image}")

        qr_reader = QRReader()

        detections = qr_reader(image)

        assert len(detections) > 0

        assert detections[0].text == expected_text


def test_empty_image():
    """
    This test verifies that the QRReader handles an empty image.
    """

    empty_image = Image.open("tests/shared_data/images/empty.png")

    qr_reader = QRReader()

    detections = qr_reader(empty_image)

    assert len(detections) == 0
