# Install all dependencies
# Check if it is macos runtime or ubuntu
# Assuming this is reun inside a python3 devcontainer
set -e # exit on error

sudo apt-get update && sudo apt-get upgrade
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
curl -sSL https://install.python-poetry.org | python -
poetry run pip install --upgrade pip setuptools toml flash_attn
if [[ "$(uname)" == "Darwin" ]]; then
	brew install zbar
else
	sudo apt update
	sudo apt-get install -y libzbar0
fi
echo "Installing poetry dependencies"
poetry lock --no-update
poetry install -E all
# POETRY run install-dependencies
echo "Installing poetry dependencies done 🎉"

echo "Installing Flash Attention"
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE poetry run pip install flash-attn --no-build-isolation
echo "Installing Flash Attention done 🎉"
