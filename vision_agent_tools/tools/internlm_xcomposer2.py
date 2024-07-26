from typing import Union

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class InternLMXComposer2:
    HF_MODEL = "internlm/internlm-xcomposer2d5-7b"

    def __init__(self, max_size = 1024):
        self.load_image = load_image
        self.load_video = get_class_from_dynamic_module(
            "ixc_utils.load_video", self.HF_MODEL
        )
        self.frame2img = get_class_from_dynamic_module(
            "ixc_utils.frame2img", self.HF_MODEL
        )
        self.video_transform = get_class_from_dynamic_module(
            "ixc_utils.Video_transform", self.HF_MODEL
        )
        self.get_font = get_class_from_dynamic_module(
            "ixc_utils.get_font", self.HF_MODEL
        )

        engine_config = TurbomindEngineConfig(model_format="awq")
        self.pipe = pipeline(self.HF_MODEL + "-4bit", backend_config=engine_config)
        self.gen_config = GenerationConfig(top_k=0, top_p=0.8, temperature=0.1)
        self.max_size = max_size

    def check_valid_image(self, file_name: str):
        return file_name.endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ) or file_name.startswith("data:image")

    def check_valid_video(self, file_name: str):
        return file_name.endswith((".mp4", ".avi", ".mov"))

    def transform_image(self, image: Image.Image):
        image = image.convert("RGB")
        if image.size[0] > self.max_size or image.size[1] > self.max_size:
            image.thumbnail((self.max_size, self.max_size))
        return image

    def __call__(
        self,
        media: Union[str, Image.Image],
        prompt: str,
    ) -> str:
        if isinstance(media, Image.Image):
            image = self.transform_image(media)
        elif self.check_valid_image(media):
            image = self.transform_image(self.load_image(media))
        elif self.check_valid_video(media):
            video = self.load_video(media)
            video = [self.transform_image(image) for image in video]
            if len(video) > 100:
                raise ValueError("Video is too long")
            image = self.frame2img(video, self.get_font())
            image = self.video_transform(image)
        else:
            raise ValueError("Invalid media type")

        sess = self.pipe.chat((prompt, image), gen_config=self.gen_config)
        return sess.response.text


if __name__ == "__main__":
    model = InternLMXComposer2()
    # print(model("section1_small.mp4", "describe this video."))
    # print(model("saved_frames/frame_0.jpg", "describe this image"))
    print(model("section1_chunk_24_32.mp4", "what is the player number who makes the kick?"))
