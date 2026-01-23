# Development Environment Setup

This document provides step-by-step instructions for setting up the development environment for the insurance underwriting LLM project.

## Prerequisites

- Python 3.12 or higher
- Git
- CUDA-compatible GPU (recommended: T4 16GB or better)

## Setup Instructions

### 1. Install uv (Python Package Manager)

```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your shell or source the configuration
source $HOME/.cargo/env
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd insurance-underwriting-llm
```

### 3. Create Virtual Environment

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the virtual environment
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Install all project dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 5. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality
pre-commit install

# Optional: Run pre-commit on all files to verify setup
pre-commit run --all-files
```

### 6. Verify CUDA Setup (If Using GPU)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

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
├── data/                  # Data directory (gitignored)
├── models/                # Saved models (gitignored)
├── results/               # Evaluation results
├── docs/                  # Documentation
├── pyproject.toml         # Project dependencies and configuration
└── README.md              # Project overview
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_data.py
```

### Code Formatting and Linting

```bash
# Format code with ruff
ruff format .

# Check for linting issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Running Notebooks

```bash
# Start Jupyter server
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

## Troubleshooting

### CUDA/GPU Issues

If you encounter CUDA-related errors:

1. Verify CUDA toolkit is installed
2. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.version.cuda)"`
3. Reinstall PyTorch with correct CUDA version if needed

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Remove virtual environment
rm -rf .venv

# Recreate environment
uv venv --python 3.12
source .venv/bin/activate

# Reinstall dependencies
uv pip install -e ".[dev]"
```

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Reinstall hooks
pre-commit uninstall
pre-commit install
```

## Next Steps

After setting up the environment:

1. Review the project documentation in `docs/`
2. Explore the data acquisition notebook: `notebooks/01_data_exploration.ipynb`
3. Run the test suite to verify everything works: `pytest`
4. Start with Phase 1 user stories in `scratchboard/phase1-user-stories.md`

## Additional Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
