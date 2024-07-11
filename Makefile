SHELL := /bin/bash

all: install test

install:
	# Install all dependencies
	pip install -r requirements.txt

test:
	# Run all unit tests
	pytest tests

serve/docs:
	# Start the documentation server
mkdocs serve

build/docs:
	# Builds the documentation
	mkdocs build -d site
