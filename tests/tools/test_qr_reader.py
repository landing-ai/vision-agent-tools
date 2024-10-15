import pytest
from PIL import Image
from vision_agent_tools.tools.qr_reader import QRReader


@pytest.mark.parametrize("expected_text", ["This is a tes"])
def test_successful_qr_tool(expected_text):
    """
    This test verifies that the QR Tool successfully detects a known QR code in an image.
    """

    for test_image in ["001.jpeg", "002.jpeg"]:
        image = Image.open(f"tests/shared_data/images/{test_image}")

        tool = QRReader(model="qr_reader")

        detection = tool(image=image)

        assert detection.text == expected_text
