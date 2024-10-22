import pytest
from PIL import Image

from vision_agent_tools.shared_types import PromptTask
from vision_agent_tools.tools.ocr import OCR, OCRModel


def test_ocr_with_florence2(shared_tool):
    image_path = "tests/shared_data/images/handwritten-text.png"
    image = Image.open(image_path)

    # default task is <OCR>
    output = shared_tool(images=[image])

    assert output == [{"text": "This is a handwrittenexampleWrite as good as you can."}]


def test_ocr_with_region_with_florence2(shared_tool):
    image_path = "tests/shared_data/images/handwritten-text.png"
    image = Image.open(image_path)

    output = shared_tool(images=[image], task=PromptTask.OCR_WITH_REGION)

    assert output == [
        {
            "labels": [
                "</s>This is a handwritten",
                "example",
                "Write as good as you can.",
            ],
            "quad_boxes": [
                [
                    30.33650016784668,
                    43.84700012207031,
                    591.7445068359375,
                    43.84700012207031,
                    591.7445068359375,
                    91.76900482177734,
                    30.33650016784668,
                    91.76900482177734,
                ],
                [
                    40.570499420166016,
                    144.2550048828125,
                    233.55450439453125,
                    143.27700805664062,
                    233.55450439453125,
                    182.0709991455078,
                    40.570499420166016,
                    183.0489959716797,
                ],
                [
                    40.570499420166016,
                    236.51300048828125,
                    686.7745361328125,
                    241.4029998779297,
                    686.7745361328125,
                    284.1090087890625,
                    40.570499420166016,
                    280.1969909667969,
                ],
            ],
        }
    ]


def test_ocr_with_florence2_invalid_task(shared_tool):
    image_path = "tests/shared_data/images/handwritten-text.png"
    image = Image.open(image_path)

    with pytest.raises(ValueError) as exc:
        shared_tool(images=[image], task=PromptTask.OBJECT_DETECTION)
        assert (
            "Invalid task: <OD>. Supported tasks are: ['<OCR>', '<OCR_WITH_REGION>']"
            in str(exc)
        )


@pytest.fixture(scope="module")
def shared_tool():
    return OCR(model=OCRModel.FLORENCE2)
