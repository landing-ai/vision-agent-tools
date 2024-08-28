import os
import platform
import subprocess


def install_dependencies():
    # Install all Poetry dependencies
    subprocess.run(["poetry", "install"], check=True)

    # List of optional dependencies
    optional_dependencies = [
        "qreader",
        "transformers",
        "scipy",
        "timm",
        "einops",
        "loca",
        "depth-anything-v2",
        "controlnet-aux",
        "lmdeploy",
        "decord",
        "sam-2",
    ]

    # Install optional dependencies
    for dep in optional_dependencies:
        subprocess.run(["poetry", "add", f"--optional", dep], check=True)

    # Conditional installation for CUDA
    if platform.system() != "Darwin":  # Skip for macOS
        subprocess.run(
            ["poetry", "add", "nvidia-cuda-runtime-cu12==12.1.105"], check=True
        )


if __name__ == "__main__":
    install_dependencies()
