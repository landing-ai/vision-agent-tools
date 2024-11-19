from vision_agent_tools.helpers.filters import filter_bbox_predictions


def test_filter_bbox_predictions_remove_big_box():
    predictions = {"bboxes": [[0, 0, 10, 10]], "labels": [0]}
    new_predictions = filter_bbox_predictions(predictions, (10, 10))
    assert new_predictions == {"bboxes": [], "labels": []}
    # the result didn't change the original predictions
    assert new_predictions != predictions


def test_filter_bbox_predictions_dont_remove_big_box():
    predictions = {"bboxes": [[0, 0, 10, 10]], "labels": [0]}
    new_predictions = filter_bbox_predictions(predictions, (100, 100))
    assert new_predictions == {"bboxes": [[0, 0, 10, 10]], "labels": [0]}


def test_filter_bbox_predictions():
    input_data = {
        "bboxes": [
            [1.5, 253.50001525878906, 546.300048828125, 599.1000366210938],
            [
                240.3000030517578,
                554.1000366210938,
                254.70001220703125,
                599.1000366210938,
            ],
            [526.5, 367.5, 542.7000122070312, 402.3000183105469],
            [
                311.1000061035156,
                453.9000244140625,
                327.3000183105469,
                483.9000244140625,
            ],
            [
                192.90000915527344,
                329.1000061035156,
                212.10000610351562,
                353.70001220703125,
            ],
            [
                351.3000183105469,
                428.70001220703125,
                365.1000061035156,
                461.70001220703125,
            ],
            [312.3000183105469, 552.300048828125, 330.3000183105469, 576.9000244140625],
            [0.30000001192092896, 256.5, 15.300000190734863, 282.3000183105469],
            [
                228.90000915527344,
                440.1000061035156,
                242.70001220703125,
                471.3000183105469,
            ],
            [
                128.10000610351562,
                383.1000061035156,
                140.10000610351562,
                417.9000244140625,
            ],
            [
                62.10000228881836,
                366.9000244140625,
                75.9000015258789,
                395.70001220703125,
            ],
            [
                251.70001220703125,
                359.1000061035156,
                263.1000061035156,
                392.1000061035156,
            ],
            [35.10000228881836, 307.5, 52.500003814697266, 328.5],
            [
                155.10000610351562,
                452.70001220703125,
                167.70001220703125,
                479.70001220703125,
            ],
            [
                204.3000030517578,
                279.9000244140625,
                223.50001525878906,
                296.1000061035156,
            ],
        ],
        "labels": [
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
        ],
    }

    filtered_data = filter_bbox_predictions(input_data, (600, 600))

    # it removed the first bbox
    assert filtered_data == {
        "bboxes": [
            [
                240.3000030517578,
                554.1000366210938,
                254.70001220703125,
                599.1000366210938,
            ],
            [526.5, 367.5, 542.7000122070312, 402.3000183105469],
            [
                311.1000061035156,
                453.9000244140625,
                327.3000183105469,
                483.9000244140625,
            ],
            [
                192.90000915527344,
                329.1000061035156,
                212.10000610351562,
                353.70001220703125,
            ],
            [
                351.3000183105469,
                428.70001220703125,
                365.1000061035156,
                461.70001220703125,
            ],
            [312.3000183105469, 552.300048828125, 330.3000183105469, 576.9000244140625],
            [0.30000001192092896, 256.5, 15.300000190734863, 282.3000183105469],
            [
                228.90000915527344,
                440.1000061035156,
                242.70001220703125,
                471.3000183105469,
            ],
            [
                128.10000610351562,
                383.1000061035156,
                140.10000610351562,
                417.9000244140625,
            ],
            [
                62.10000228881836,
                366.9000244140625,
                75.9000015258789,
                395.70001220703125,
            ],
            [
                251.70001220703125,
                359.1000061035156,
                263.1000061035156,
                392.1000061035156,
            ],
            [35.10000228881836, 307.5, 52.500003814697266, 328.5],
            [
                155.10000610351562,
                452.70001220703125,
                167.70001220703125,
                479.70001220703125,
            ],
            [
                204.3000030517578,
                279.9000244140625,
                223.50001525878906,
                296.1000061035156,
            ],
        ],
        "labels": [
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "sheep",
        ],
    }


def test_filter_valid_bboxes():
    predictions = {
        "bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
        "labels": ["sheep", "sheep"],
    }
    image_size = (100, 100)

    results = filter_bbox_predictions(predictions, image_size)
    assert results == predictions


def test_filter_wrong_order():
    predictions = {
        "bboxes": [[326.0, 362.6, 225.8, 262.9], [50, 60, 70, 80]],
        "labels": ["sheep", "sheep"],
    }
    image_size = (835, 453)

    results = filter_bbox_predictions(predictions, image_size)
    assert results == {"bboxes": [[50, 60, 70, 80]], "labels": ["sheep"]}


def test_filter_invalid_bboxes_negative_coords():
    predictions = {
        "bboxes": [[-10, 20, 30, 40], [50, 60, 70, 80]],
        "labels": ["sheep", "sheep"],
    }
    image_size = (100, 100)

    results = filter_bbox_predictions(predictions, image_size)
    assert results == {"bboxes": [[50, 60, 70, 80]], "labels": ["sheep"]}


def test_filter_invalid_bboxes_out_of_bounds():
    predictions = {
        "bboxes": [[10, 20, 110, 40], [50, 60, 70, 80]],
        "labels": ["sheep", "sheep"],
    }
    image_size = (100, 100)

    results = filter_bbox_predictions(predictions, image_size)
    assert results == {"bboxes": [[50, 60, 70, 80]], "labels": ["sheep"]}


def test_filter_invalid_bboxes_mixed_valid_invalid():
    predictions = {
        "bboxes": [
            [10, 20, 30, 40],
            [-10, 20, 30, 40],
            [50, 60, 70, 80],
            [110, 20, 120, 40],
        ],
        "labels": ["sheep", "sheep"],
    }
    image_size = (100, 100)

    results = filter_bbox_predictions(predictions, image_size)
    assert results == {
        "bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
        "labels": ["sheep"],
    }


def test_filter_invalid_bboxes():
    predictions_list = [
        {"bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]], "labels": ["sheep", "sheep"]},
        {"bboxes": [[-10, 20, 30, 40], [50, 60, 70, 80]], "labels": ["sheep", "sheep"]},
        {"bboxes": [[10, 20, 110, 40], [50, 60, 70, 80]], "labels": ["sheep", "sheep"]},
        {"bboxes": [[10, 20, 30, 40], [50, 60, 40, 70]], "labels": ["sheep", "sheep"]},
        {"bboxes": [[10, 20, 30, 40], [50, 60, 70, 60]], "labels": ["sheep", "sheep"]},
    ]

    image_size = (100, 100)

    expected_results = [
        {"bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]], "labels": ["sheep", "sheep"]},
        {"bboxes": [[50, 60, 70, 80]], "labels": ["sheep"]},
        {"bboxes": [[50, 60, 70, 80]], "labels": ["sheep"]},
        {"bboxes": [[10, 20, 30, 40]], "labels": ["sheep"]},
        {"bboxes": [[10, 20, 30, 40]], "labels": ["sheep"]},
    ]

    for idx, prediction in enumerate(predictions_list):
        results = filter_bbox_predictions(prediction, image_size)
        assert results == expected_results[idx]
