from PIL import Image

from vision_agent_tools.tools.ocr import OCR


def test_successful_ocr():
    test_image = "handwritten-text.png"

    image = Image.open(f"tests/shared_data/images/{test_image}")

    tool = OCR(model="florence2")
    output = tool(image=image)

    assert output == "This is a handwrittenexampleWrite as good as you can."
