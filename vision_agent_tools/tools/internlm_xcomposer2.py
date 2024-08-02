from typing import Union

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vision_agent_tools.tools.shared_types import BaseTool


def check_valid_image(file_name: str):
    return file_name.endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ) or file_name.startswith("data:image")


def check_valid_video(file_name: str):
    return file_name.endswith((".mp4", ".avi", ".mov"))


def transform_image(image: Image.Image, max_size: int):
    image = image.convert("RGB")
    if image.size[0] > max_size or image.size[1] > max_size:
        image.thumbnail((max_size, max_size))
    return image


class InternLMXComposer2(BaseTool):
    HF_MODEL = "internlm/internlm-xcomposer2d5-7b"

    def __init__(self, max_size=1024):
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

        engine_config = TurbomindEngineConfig(
            model_format="awq", cache_max_entry_count=0.2
        )
        self.pipe = pipeline(self.HF_MODEL + "-4bit", backend_config=engine_config)
        self.gen_config = GenerationConfig(top_k=0, top_p=0.8, temperature=0.1)
        self.max_size = max_size

    def __call__(
        self,
        media: Union[str, Image.Image],
        prompt: str,
    ) -> str:
        if isinstance(media, Image.Image):
            image = transform_image(media, self.max_size)
        elif check_valid_image(media):
            image = transform_image(self.load_image(media), self.max_size)
        elif check_valid_video(media):
            video = self.load_video(media)
            video = [transform_image(image, self.max_size) for image in video]
            # 32 frames is a ~60s video clip
            if len(video) > 32:
                raise ValueError("Video is too long")
            image = self.frame2img(video, self.get_font())
            image = self.video_transform(image)
        else:
            raise ValueError("Invalid media type")

        sess = self.pipe.chat((prompt, image), gen_config=self.gen_config)
        return sess.response.text
