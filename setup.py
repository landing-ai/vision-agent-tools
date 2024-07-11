import os
from setuptools import setup, find_packages
from itertools import chain


def read(filename):
    with open(filename, 'r') as file_handle:
        return file_handle.read()
README = os.path.join(os.path.dirname(__file__), 'README.md')

REQUIRES = ["pydantic", "numpy", "pillow", "gdown", "wget", "torch==2.2.2"]
EXTRAS_REQUIRE = {
    "qr-reader":  ["qreader"],
    "owlv2":  ["transformers", "scipy"],
    "florencev2":  [
        "transformers",
        "scipy", "timm",
        "flash-attn @ git+https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
      ],
    "loca-model": ["loca @ git+https://github.com/landing-ai/loca.git"],
    "depth-anything-v2-model": ["depth-anything-v2 @ git+https://github.com/landing-ai/depth-anything-v2.git"],
}

EXTRAS_REQUIRE['all'] = list(set(chain(*EXTRAS_REQUIRE.values())))


setup(     
    name="vision-agent-tools",     
    version="0.1.0",
    description = "Toolbox for vision tasks",
    long_description=read(README),
    long_description_content_type="text/markdown",
    author = "Landing AI",
    author_email = "dev@landing.ai",
    url="https://github.com/landing-ai/vision-agent-tools",
    python_requires=">=3.9",   
    packages=find_packages('.'),
    install_requires=REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)