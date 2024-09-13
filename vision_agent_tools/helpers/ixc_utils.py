# File copied from https://huggingface.co/internlm/internlm-xcomposer2d5-7b/raw/main/ixc_utils.py
# This copy was required to remove the full dependency from decord to load videos
import numpy as np

from urllib.request import urlopen
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


def get_font():
    truetype_url = "https://huggingface.co/internlm/internlm-xcomposer2d5-7b/resolve/main/SimHei.ttf?download=true"
    ff = urlopen(truetype_url)
    font = ImageFont.truetype(ff, size=40)
    return font


def Video_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = width / height
    scale = 1
    new_h = int(scale * 560)
    new_w = int(new_h * ratio)
    # print (new_h, new_w)

    img = transforms.functional.resize(
        img,
        [new_h, new_w],
    )
    img = img.transpose(Image.TRANSPOSE)
    img = _padding_336(img, 560)
    width, height = img.size
    if not trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def frame2img(imgs, font):
    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w / h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = transforms.functional.resize(
            img,
            [new_h, new_w],
        )
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w, h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        new_img = Image.new("RGB", (new_w, new_h), "white")
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h), f"<IMAGE {idx}>", font=font, fill="black")
            if idx + 1 < len(imgs):
                draw.line(
                    [(0, pad + curr_h + h + 5), (new_w, pad + curr_h + h + 5)],
                    fill="black",
                    width=2,
                )
            curr_h += h + 10 + pad
        # print (new_w, new_h)
    else:
        for im in imgs:
            w, h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        new_img = Image.new("RGB", (new_w, new_h), "white")
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f"<IMAGE {idx}>", font=font, fill="black")
            if idx + 1 < len(imgs):
                draw.line(
                    [(curr_w + w + 5, 0), (curr_w + w + 5, new_h)],
                    fill="black",
                    width=2,
                )
            curr_w += w + 10
    return new_img


def _padding_336(b, pad=336):
    width, height = b.size
    tar = int(np.ceil(height / pad) * pad)
    top_padding = 0  # int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(
        b,
        [left_padding, top_padding, right_padding, bottom_padding],
        fill=[255, 255, 255],
    )

    return b
