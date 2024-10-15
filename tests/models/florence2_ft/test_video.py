from model.video import bytes_to_pil


def test_bytes_to_pil():
    video_path = "tests/data/shark_10fps.mp4"
    with open(video_path, "rb") as file:
        video = file.read()
        result = bytes_to_pil(video)

    assert len(result) == 80
    for frame in result:
        assert frame.size == (1920, 1080)
