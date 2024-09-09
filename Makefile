SHELL := /bin/bash
POETRY := poetry

all: install test

install:
	bash ./scripts/install.sh

test:
	$(POETRY) run pytest tests/tools/test_text_to_object_detection.py
	$(POETRY) run pytest tests

serve/docs:
	# Start the documentation server
	$(POETRY) run mkdocs serve

build/docs:
	# Builds the documentation
	$(POETRY) run mkdocs build -d site
