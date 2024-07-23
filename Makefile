SHELL := /bin/bash
POETRY := poetry

all: install test

install:
	# Install all dependencies
	$(POETRY) install -E all
	pip install flash-attn

install-qr-reader:
	# Install qr-reader dependencies only
	sudo apt update
	sudo apt-get install -y libzbar0
	$(POETRY) install -E qr-reader --no-interaction

install-owlv2:
	# Install owlv2 dependencies only
	$(POETRY) install -E owlv2 --no-interaction

install-zeroshot-counting:
	# Install loca dependencies only
	$(POETRY) install -E loca-model

install-depth-anything-v2:
	# Install depth-anything-v2 dependencies only
	$(POETRY) install -E depth-anything-v2-model --no-interaction

install-florencev2:
	# Install florencev2 dependencies only
	$(POETRY) install -E florencev2 --no-interaction
	pip install flash-attn

install-nsfw-classification:
	# Install nsfw_classification dependencies only
	$(POETRY) install -E nsfw-classification --no-interaction

install-controlnet-aux:
	# Install controlnet_aux dependencies only
	$(POETRY) install -E controlnet-aux --no-interaction

install-florencev2-qa:
	# Install florencev2-qa dependencies only
	$(POETRY) install -E florencev2-qa --no-interaction
	pip install flash-attn

test:
	# Run all unit tests (experimental due posible dependencies conflict)
	$(POETRY) run pytest tests

serve/docs:
	# Start the documentation server
	$(POETRY) run mkdocs serve

build/docs:
	# Builds the documentation
	$(POETRY) run mkdocs build -d site

test-qr-reader:
	# Run qr-reader unit tests
	$(POETRY) run pytest tests/tools/test_qr_reader.py

test-owlv2:
	# Run owlv2 unit tests
	$(POETRY) run pytest tests/tools/test_owlv2.py

test-zeroshot-counting:
	# Run zeroshot-counting unit tests
	$(POETRY) run pytest tests/tools/test_loca.py

test-depth-anything-v2:
	# Run depth-anything-v2 unit tests
	$(POETRY) run pytest tests/tools/test_depth_anything_v2.py

test-florencev2:
	# Run florencev2 unit tests
	$(POETRY) run pytest tests/tools/test_florencev2.py

test-nsfw-classification:
	# Run nsfw_classification unit tests
	$(POETRY) run pytest tests/tools/test_nsfw_classification.py

test-controlnet-aux:
	# Run controlnet_aux unit tests
	$(POETRY) run pytest tests/tools/test_controlnet_aux.py

test-florencev2-qa:
	# Run florencev2-qa unit tests
	$(POETRY) run pytest tests/tools/test_florencev2_qa.py
