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
	@echo ""
	@echo "Data & Model Setup:"
	@echo "  make download-data    - Download datasets"
	@echo "  make preprocess       - Preprocess datasets"
	@echo "  make setup-models     - Download pretrained models (no fine-tuning, ready to use)"
	@echo "  make finetune-all     - Fine-tune all models on your datasets (requires data)"
	@echo ""
	@echo "Model Comparison:"
	@echo "  make compare          - Compare all 6 models (50 samples, ~10-20 min)"
	@echo "  make compare-quick    - Quick comparison (10 samples, extractive only, ~2 min)"
	@echo "  make compare-full     - Full comparison (100 samples, all metrics, ~30-60 min)"
	@echo ""
	@echo "Length Analysis:"
	@echo "  make analyze-length        - Analyze performance vs document length (100 samples, ~20-30 min)"
	@echo "  make analyze-length-quick  - Quick length analysis (50 samples, ~10 min)"
	@echo "  make analyze-length-full   - Full length analysis (200 samples, ~60-90 min)"
	@echo ""
	@echo "Section Analysis:"
	@echo "  make analyze-sections       - Analyze section structure & coverage (50 samples, ~15-20 min)"
	@echo "  make analyze-sections-quick - Quick section analysis (30 samples, ~5-10 min)"
	@echo ""
	@echo "Demo:"
	@echo "  make demo             - Launch Streamlit demo"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo ""
	@echo "Utilities:"
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

setup-models:
	@echo "Setting up pretrained models (no fine-tuning)..."
	@echo "This will download pretrained models but skip fine-tuning."
	@echo ""
	@# Temporarily enable skip_fine_tuning for all configs
	@sed -i.bak 's/skip_fine_tuning: false/skip_fine_tuning: true/' configs/baseline.yaml configs/hierarchical.yaml configs/longformer.yaml
	@$(PYTHON_VENV) src/training.py --config configs/baseline.yaml
	@$(PYTHON_VENV) src/training.py --config configs/hierarchical.yaml
	@$(PYTHON_VENV) src/training.py --config configs/longformer.yaml
	@# Restore original configs
	@mv configs/baseline.yaml.bak configs/baseline.yaml 2>/dev/null || true
	@mv configs/hierarchical.yaml.bak configs/hierarchical.yaml 2>/dev/null || true
	@mv configs/longformer.yaml.bak configs/longformer.yaml 2>/dev/null || true
	@echo ""
	@echo "✓ Pretrained models are ready to use!"
	@echo "To fine-tune models on your datasets, run: make finetune-all"

finetune-all:
	@echo "Fine-tuning all models on datasets..."
	@echo ""
	@echo "WARNING: This requires:"
	@echo "  - Preprocessed datasets in data/processed/"
	@echo "  - ~4GB disk space for model downloads"
	@echo "  - GPU recommended for training"
	@echo "  - Several hours to complete"
	@echo ""
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		sed -i.bak 's/skip_fine_tuning: true/skip_fine_tuning: false/' configs/baseline.yaml configs/hierarchical.yaml configs/longformer.yaml && \
		$(PYTHON_VENV) src/training.py --config configs/baseline.yaml && \
		$(PYTHON_VENV) src/training.py --config configs/hierarchical.yaml && \
		$(PYTHON_VENV) src/training.py --config configs/longformer.yaml && \
		mv configs/baseline.yaml.bak configs/baseline.yaml 2>/dev/null || true && \
		mv configs/hierarchical.yaml.bak configs/hierarchical.yaml 2>/dev/null || true && \
		mv configs/longformer.yaml.bak configs/longformer.yaml 2>/dev/null || true; \
		echo ""; \
		echo "✓ Fine-tuning complete! Models saved in models/checkpoints/"; \
	else \
		echo "Cancelled."; \
	fi

# Legacy alias (deprecated - use setup-models instead)
train-baseline:
	@echo "WARNING: 'train-baseline' is deprecated. Use 'make setup-models' instead."
	@$(MAKE) setup-models


compare:
	@echo "Comparing all models..."
	@echo ""
	@echo "This will:"
	@echo "  - Run all 6 models (TextRank, LexRank, BART, Hierarchical, Longformer, Sliding Window)"
	@echo "  - Evaluate on test dataset"
	@echo "  - Generate comparison tables and visualizations"
	@echo "  - Create comprehensive report"
	@echo ""
	@if [ ! -d "data/processed/arxiv" ]; then \
		echo "ERROR: No processed data found. Please run 'make preprocess' first."; \
		echo ""; \
		echo "Quick start:"; \
		echo "  1. make download-data  (download datasets)"; \
		echo "  2. make preprocess     (prepare datasets)"; \
		echo "  3. make compare        (run comparison)"; \
		exit 1; \
	fi
	@echo "Starting comparison on 50 samples..."
	@echo ""
	$(PYTHON_VENV) scripts/compare_models.py \
		--dataset arxiv \
		--num-samples 50 \
		--output-dir results/comparison \
		--device cpu
	@echo ""
	@echo "✓ Comparison complete!"
	@echo ""
	@echo "View results:"
	@echo "  Report:  results/comparison/comparison_report.md"
	@echo "  Plots:   results/comparison/*.png"
	@echo "  Data:    results/comparison/comparison_results.csv"

compare-quick:
	@echo "Running quick comparison (10 samples, extractive models only)..."
	$(PYTHON_VENV) scripts/compare_models.py \
		--dataset arxiv \
		--num-samples 10 \
		--output-dir results/comparison_quick \
		--models textrank lexrank \
		--metrics rouge \
		--device cpu

compare-full:
	@echo "Running full comparison (100 samples, all metrics)..."
	@read -p "This may take 30-60 minutes. Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON_VENV) scripts/compare_models.py \
			--dataset arxiv \
			--num-samples 100 \
			--output-dir results/comparison_full \
			--device cpu; \
	fi
analyze-length:
	@echo "Analyzing performance degradation with document length..."
	@echo ""
	@echo "This will:"
	@echo "  - Bin documents by length (6K, 9K, 12K, 15K+ tokens)"
	@echo "  - Run all models on each length bin"
	@echo "  - Analyze degradation rates"
	@echo "  - Generate visualizations and report"
	@echo ""
	@if [ ! -d "data/processed/arxiv" ]; then \
		echo "ERROR: No processed data found. Please run 'make preprocess' first."; \
		exit 1; \
	fi
	@echo "Starting length analysis (100 samples)..."
	@echo ""
	$(PYTHON_VENV) scripts/analyze_length_degradation.py \
		--dataset arxiv \
		--num-samples 100 \
		--output-dir results/length_analysis \
		--device cpu
	@echo ""
	@echo "✓ Length analysis complete!"
	@echo ""
	@echo "View results:"
	@echo "  Report:  results/length_analysis/length_analysis_report.md"
	@echo "  Plots:   results/length_analysis/*.png"
	@echo "  Data:    results/length_analysis/length_analysis_results.json"

analyze-length-quick:
	@echo "Running quick length analysis (50 samples, extractive only)..."
	$(PYTHON_VENV) scripts/analyze_length_degradation.py \
		--dataset arxiv \
		--num-samples 50 \
		--output-dir results/length_analysis_quick \
		--models textrank lexrank \
		--metrics rouge \
		--device cpu

analyze-length-full:
	@echo "Running comprehensive length analysis (200 samples)..."
	@read -p "This may take 60-90 minutes. Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON_VENV) scripts/analyze_length_degradation.py \
			--dataset arxiv \
			--num-samples 200 \
			--output-dir results/length_analysis_full \
			--device cpu; \
	fi

analyze-sections:
	@echo "Analyzing document section structure..."
	@echo ""
	@echo "This will:"
	@echo "  - Detect sections in documents (Introduction, Methods, etc.)"
	@echo "  - Analyze section coverage in summaries"
	@echo "  - Compare section-aware vs regular summarization"
	@echo "  - Generate visualizations and report"
	@echo ""
	@if [ ! -d "data/processed/arxiv" ]; then \
		echo "ERROR: No processed data found. Please run 'make preprocess' first."; \
		exit 1; \
	fi
	@echo "Starting section analysis (50 samples)..."
	@echo ""
	$(PYTHON_VENV) scripts/analyze_sections.py \
		--dataset arxiv \
		--num-samples 50 \
		--output-dir results/section_analysis \
		--compare-models \
		--device cpu
	@echo ""
	@echo "✓ Section analysis complete!"
	@echo ""
	@echo "View results:"
	@echo "  Report:  results/section_analysis/section_analysis_report.md"
	@echo "  Plots:   results/section_analysis/*.png"
	@echo "  Data:    results/section_analysis/section_analysis_results.json"

analyze-sections-quick:
	@echo "Running quick section analysis (no model comparison)..."
	$(PYTHON_VENV) scripts/analyze_sections.py \
		--dataset arxiv \
		--num-samples 30 \
		--output-dir results/section_analysis_quick \
		--device cpu

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

