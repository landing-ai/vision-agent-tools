SHELL := /bin/bash
POETRY := poetry

all: install test

install:
	# Install all dependencies
	sudo apt update
	sudo apt-get install -y libzbar0
	$(POETRY) install -E all
	$(POETRY) run pip install flash-attn

test:
	$(POETRY) run pytest tests

serve/docs:
	# Start the documentation server
	$(POETRY) run mkdocs serve

build/docs:
	# Builds the documentation
	$(POETRY) run mkdocs build -d site
