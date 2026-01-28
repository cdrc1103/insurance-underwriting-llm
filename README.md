# Insurance Underwriting LLM

Multi-turn conversation fine-tuning for insurance underwriting using parameter-efficient methods (LoRA/QLoRA).

## Overview

This project demonstrates finetuning a small language model to perform multi-turn dialogue for insurance underwriting tasks. The goal is to take a base model with limited conversational abilities and adapt it to understand insurance domain knowledge and engage in coherent multi-turn conversations with underwriters.

## Problem Statement

Insurance underwriters need to assess company eligibility, recommend products, and make coverage decisions. This requires:
- Understanding company profiles (revenue, employees, location, industry)
- Knowledge of insurance products and underwriting guidelines
- Ability to engage in multi-turn conversations to gather information and provide recommendations
- Domain-specific reasoning about risk and appetite

Small pretrained models typically lack:
- Insurance domain knowledge
- Multi-turn conversational coherence
- Ability to reason about underwriting criteria

## Objectives

### Primary Goal
Finetune a small language model (< 1B parameters) to handle multi-turn insurance underwriting conversations using the Snorkel AI Multi-Turn Insurance Underwriting dataset.

### Success Criteria
1. Model can maintain context across multiple conversation turns
2. Model generates accurate underwriting recommendations based on company information
3. Model demonstrates understanding of insurance terminology and concepts
4. Measurable improvement over base model on evaluation metrics

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

## Dataset

This project uses the [Multi-Turn Insurance Underwriting dataset](https://huggingface.co/datasets/snorkelai/Multi-Turn-Insurance-Underwriting) from Hugging Face, containing ~380 multi-turn conversations about insurance underwriting scenarios.

### Dataset Splits

The dataset has been preprocessed and split into:
- **Train**: 222 examples (75%)
- **Validation**: 37 examples (12.5%)
- **Test**: 37 examples (12.5%)

Splits are stratified by task type to ensure balanced representation across all sets.

### Token Analysis

Token distribution analysis using the Qwen3-0.6B tokenizer (vocab size: 151,643):

| Split | Mean Tokens | Median Tokens | Min | Max | 95th Percentile |
|-------|------------|---------------|-----|-----|-----------------|
| Train | 5,425 | 2,442 | 472 | 64,258 | 21,486 |
| Validation | 4,087 | 2,119 | 804 | 20,415 | 16,625 |
| Test | 5,431 | 2,564 | 1,205 | 29,997 | 22,295 |

**Recommended max_length**: 21,486 tokens (based on 95th percentile of training set)

This configuration covers 95% of training examples without truncation while maintaining memory efficiency on T4 16GB GPUs. Only ~5% of examples across all splits will be truncated at this length.

For detailed token analysis, see [notebooks/03_token_analysis.ipynb](notebooks/03_token_analysis.ipynb).

## Technology Stack

- **ML Framework**: PyTorch, Hugging Face Transformers
- **Fine-tuning**: PEFT (LoRA/QLoRA), bitsandbytes
- **Evaluation**: evaluate, rouge-score, sacrebleu
- **Development**: ruff, pytest, pre-commit
