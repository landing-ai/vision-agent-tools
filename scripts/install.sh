# Install all dependencies
# Check if it is macos runtime or ubuntu
# Assuming this is reun inside a python3 devcontainer
set -e # exit on error

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
curl -sSL https://install.python-poetry.org | python -
poetry run pip install --upgrade pip setuptools toml

if [[ "$(uname)" == "Darwin" ]]; then
	brew install zbar
else
	sudo apt-get update && sudo apt-get upgrade -y
	sudo apt-get install -y libzbar0
fi

echo "Installing poetry dependencies"
poetry install -E all
# POETRY run install-dependencies
echo "Installing poetry dependencies done 🎉"
