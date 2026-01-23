# Insurance Underwriting LLM

Multi-turn conversation fine-tuning for insurance underwriting using parameter-efficient methods (LoRA/QLoRA).

## Overview

This project implements a production-grade RAG system that processes Singapore financial documents with custom fine-tuned embeddings. The system demonstrates end-to-end AI engineering capabilities including data preprocessing, model fine-tuning, and comprehensive evaluation.

### Key Features

- **Multi-turn conversation handling** for insurance underwriting scenarios
- **Parameter-efficient fine-tuning** using LoRA/QLoRA
- **Comprehensive evaluation** with quantitative and qualitative metrics
- **Production-ready code** with tests, type hints, and documentation
- **Reproducible setup** with version-controlled dependencies

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended: T4 16GB or better)
- Git

### Installation

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd insurance-underwriting-llm

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

For detailed setup instructions, see [docs/setup.md](docs/setup.md).

## Project Structure

```
insurance-underwriting-llm/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model configuration and loading
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Helper functions
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Executable scripts
├── configs/               # Configuration files
├── docs/                  # Documentation
└── scratchboard/          # Planning and user stories
```

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_data.py -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Run pre-commit hooks
pre-commit run --all-files
```

## Project Phases

This project is organized into 5 phases:

1. **Phase 1: Environment & Data Setup** - Infrastructure setup, data acquisition, and preprocessing
2. **Phase 2: Baseline Evaluation** - Zero-shot and few-shot evaluation of base model
3. **Phase 3: Model Finetuning** - LoRA/QLoRA fine-tuning implementation
4. **Phase 4: Evaluation & Analysis** - Comprehensive performance analysis
5. **Phase 5: Documentation & Showcase** - Polish and portfolio preparation

See `scratchboard/` directory for detailed user stories for each phase.

## Dataset

This project uses the [Multi-Turn Insurance Underwriting dataset](https://huggingface.co/datasets/snorkelai/Multi-Turn-Insurance-Underwriting) from Hugging Face, containing ~380 multi-turn conversations about insurance underwriting scenarios.

## Technology Stack

- **ML Framework**: PyTorch, Hugging Face Transformers
- **Fine-tuning**: PEFT (LoRA/QLoRA), bitsandbytes
- **Evaluation**: evaluate, rouge-score, sacrebleu
- **Development**: ruff, pytest, pre-commit
- **Notebooks**: Jupyter, matplotlib, seaborn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This is a personal portfolio project. For detailed development guidelines, see [CLAUDE.md](CLAUDE.md).

## Contact

For questions or feedback, please open an issue on GitHub.
