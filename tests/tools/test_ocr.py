from PIL import Image

from vision_agent_tools.tools.ocr import OCR


def test_successful_ocr():
    test_image = "handwritten-text.png"

    image = Image.open(f"tests/tools/data/ocr/{test_image}")

    tool = OCR(model="florencev2")
    output = tool(image=image)

    assert output == "This is a handwrittenexampleWrite as good as you can."
