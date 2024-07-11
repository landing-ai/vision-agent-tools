import os
from setuptools import setup, find_packages


def read(filename):
    with open(filename, 'r') as file_handle:
        return file_handle.read()
README = os.path.join(os.path.dirname(__file__), 'README.md')

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
)