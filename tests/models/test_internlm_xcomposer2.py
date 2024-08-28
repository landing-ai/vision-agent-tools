from PIL import Image
from vision_agent_tools.models.internlm_xcomposer2 import InternLMXComposer2


def test_successful_internlm_xcomposer2_for_video(random_video_generator):
    video_np = random_video_generator()
    prompt = "Here are some frames of a video. Describe this video in detail"

    run_inference = InternLMXComposer2()

    answer = run_inference(video=video_np, prompt=prompt)

    assert len(answer) > 0


def test_successful_internlm_xcomposer2_for_video_chunks(random_video_generator):
    video_np = random_video_generator(n_frames=10)
    prompt = "Here are some frames of a video. Describe this video in detail"

    run_inference = InternLMXComposer2()

    answer = run_inference(video=video_np, prompt=prompt, n_chunks=2)

    assert len(answer) == 2


def test_successful_internlm_xcomposer2_for_images():
    test_image = "car.jpg"

    image = Image.open(f"tests/tools/data/florencev2/{test_image}")

    run_inference = InternLMXComposer2()

    answer = run_inference(image=image, prompt="what is the color of the car?")

    assert len(answer) > 0
