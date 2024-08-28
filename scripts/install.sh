# Install all dependencies
# Check if it is macos runtime or ubuntu
set -e # exit on error

# Poetry should be > 2.1
POETRY run pip install --upgrade pip setuptools
if [[ "$(uname)" == "Darwin" ]]; then
	brew install zbar
else
	sudo apt update
	sudo apt-get install -y libzbar0
fi
POETRY run install-dependencies
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE POETRY run pip install flash-attn --no-build-isolation
