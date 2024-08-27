import pytest
from PIL import Image
from vision_agent_tools.tools.ocr import OCR


@pytest.mark.parametrize("expected_text", ["This is a tes"])
def test_successful_ocr_tool(expected_text):
    """
    This test verifies that the OCR Tool successfully detects a known QR code in an image.
    """

    for test_image in ["001.jpeg", "002.jpeg"]:
        image = Image.open(f"tests/tools/data/qr_reader/{test_image}")

        ocr_tool = OCR(model="qr_reader")

        detection = ocr_tool(image=image)

        assert detection.text == expected_text
