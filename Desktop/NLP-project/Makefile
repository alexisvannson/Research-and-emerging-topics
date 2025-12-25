.PHONY: help install setup lint format test download-data preprocess train-baseline train-all evaluate demo docker-build docker-run clean

PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python
PROJECT_ROOT := $(shell pwd)
export PYTHONPATH := $(PROJECT_ROOT)

help:
	@echo "Available targets:"
	@echo "  make install          - Install dependencies in virtual environment"
	@echo "  make setup            - Complete project setup (venv + install + download data)"
	@echo "  make lint             - Run code quality checks (black, flake8, mypy)"
	@echo "  make format           - Auto-format code with black and isort"
	@echo "  make test             - Run unit tests with pytest"
	@echo "  make download-data    - Download datasets"
	@echo "  make preprocess       - Preprocess datasets"
	@echo "  make train-baseline   - Train baseline models"
	@echo "  make train-all        - Train all models"
	@echo "  make evaluate         - Run evaluation suite"
	@echo "  make demo             - Launch Streamlit demo"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make clean            - Remove generated files and cache"


install:
	@echo "Installing dependencies..."
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install "numpy<2"
	$(PIP) install -r requirements.txt
	$(PYTHON_VENV) -m spacy download en_core_web_sm
	$(PYTHON_VENV) -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
	@echo "Dependencies installed successfully!"

setup: install
	@echo "Setting up project..."
	mkdir -p data/raw data/processed logs models/checkpoints
	@echo "Downloading datasets..."
	$(PYTHON_VENV) data/scripts/download_datasets.py
	@echo "Setup complete!"

lint:
	@echo "Running code quality checks..."
	$(VENV_BIN)/black --check models/ src/ tests/ app.py || true
	$(VENV_BIN)/flake8 models/ src/ tests/ app.py --max-line-length=100 --extend-ignore=E203,W503 || true
	$(VENV_BIN)/mypy models/ src/ --ignore-missing-imports || true
	@echo "Lint check complete!"

format:
	@echo "Formatting code..."
	$(VENV_BIN)/isort models/ src/ tests/ app.py data/scripts/
	$(VENV_BIN)/black models/ src/ tests/ app.py data/scripts/
	@echo "Code formatted successfully!"

test:
	@echo "Running tests..."
	$(VENV_BIN)/pytest tests/ -v --cov=src --cov=models --cov-report=html --cov-report=term
	@echo "Tests complete!"

download-data:
	@echo "Downloading datasets..."
	$(PYTHON_VENV) data/scripts/download_datasets.py

preprocess:
	@echo "Preprocessing datasets..."
	$(PYTHON_VENV) data/scripts/preprocess.py

train-baseline:
	@echo "Training baseline models..."
	$(PYTHON_VENV) src/training.py --config configs/baseline.yaml

train-all:
	@echo "Training all models..."
	$(PYTHON_VENV) src/training.py --config configs/baseline.yaml
	$(PYTHON_VENV) src/training.py --config configs/hierarchical.yaml
	$(PYTHON_VENV) src/training.py --config configs/longformer.yaml

evaluate:
	@echo "Running evaluation suite..."
	$(PYTHON_VENV) src/evaluation.py --all-models

demo:
	@echo "Launching Streamlit demo..."
	$(VENV_BIN)/streamlit run app.py

docker-build:
	@echo "Building Docker image..."
	docker build -t long-doc-summarization:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8501:8501 long-doc-summarization:latest

clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf build dist
	@echo "Cleanup complete!"
