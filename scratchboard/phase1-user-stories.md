# Phase 1: Environment & Data Setup - User Stories

## Overview
Phase 1 establishes the foundational infrastructure for the insurance underwriting LLM project, including environment setup, data acquisition, preprocessing, and splitting.

---

## User Story 1.1: Development Environment Setup

**As a** ML engineer
**I want to** set up a Python development environment with all required dependencies
**So that** I can begin development with a reproducible, version-controlled setup

### Acceptance Criteria
- [ ] Python 3.12+ virtual environment created using `uv`
- [ ] Core dependencies installed: `transformers`, `peft`, `datasets`, `bitsandbytes`, `accelerate`, `torch`
- [ ] Evaluation libraries installed: `evaluate`, `rouge_score`, `sacrebleu`
- [ ] Development tools configured: `ruff`, `pytest`, `pre-commit`
- [ ] `requirements.txt` or `pyproject.toml` created for dependency tracking
- [ ] Git repository initialized with proper `.gitignore`
- [ ] Environment can be recreated from scratch using documented commands

### Technical Considerations
- Target GPU: T4 16GB VRAM (Lightning AI Free Tier)
- Ensure CUDA compatibility for PyTorch and bitsandbytes
- Pin major versions to avoid breaking changes
- Consider using `uv` for faster dependency resolution

### Dependencies
- None (foundational)

---

## User Story 1.2: Dataset Acquisition and Exploration

**As a** ML engineer
**I want to** download and explore the Multi-Turn Insurance Underwriting dataset
**So that** I understand the data structure, distribution, and quality before preprocessing

### Acceptance Criteria
- [ ] Dataset downloaded from `snorkelai/Multi-Turn-Insurance-Underwriting` on Hugging Face
- [ ] Data schema documented (company fields, conversation structure, labels)
- [ ] Exploratory analysis completed:
  - Total number of examples (380 expected)
  - Task type distribution (appetite checks, product recommendations, etc.)
  - Conversation length statistics (number of turns per example)
  - Text length statistics (tokens per turn)
- [ ] Data quality issues identified (missing fields, inconsistencies)
- [ ] Findings documented in exploration notebook or markdown file

### Technical Considerations
- Use Hugging Face `datasets` library for loading
- Check for class imbalance across task types
- Use all dataset examples (including those with tool calls)
- Verify that company profiles and conversations are properly paired

### Dependencies
- User Story 1.1 (Environment Setup)

---

## User Story 1.3: Data Preprocessing Pipeline

**As a** ML engineer
**I want to** implement a preprocessing pipeline that formats multi-turn conversations
**So that** the data is ready for model training in a standardized format

### Acceptance Criteria
- [ ] Company profile extraction implemented (name, revenue, employees, industry, state)
- [ ] Multi-turn conversation extraction with role labels (underwriter/assistant)
- [ ] Tool calls and function calling segments included for agent training
- [ ] Text cleaning applied (normalize whitespace, handle special characters)
- [ ] Industry classification extracted from business descriptions
- [ ] Standardized input format implemented:
  ```
  Company Profile:
  - Name: [company_name]
  - Revenue: [annual_revenue]
  - Employees: [number_of_employees]
  - Industry: [industry]
  - State: [state]

  Conversation:
  User: [question]
  Assistant: [response]
  ...
  ```
- [ ] Preprocessing pipeline is modular and testable
- [ ] Unit tests cover edge cases (missing fields, malformed conversations)

### Technical Considerations
- Handle missing or null company fields gracefully
- Ensure conversation ordering is preserved
- Consider max sequence length constraints (1024-2048 tokens)
- Validate that all turns have proper role assignments

### Dependencies
- User Story 1.2 (Dataset Exploration)

---

## User Story 1.4: Train/Validation/Test Split Creation

**As a** ML engineer
**I want to** create stratified train/validation/test splits of the dataset
**So that** I can train models and evaluate performance on held-out data

### Acceptance Criteria
- [ ] Dataset split into train/validation/test sets (e.g., 300/40/40 examples)
- [ ] Stratification applied by task type to ensure balanced representation
- [ ] Split ratios configurable via parameters
- [ ] Random seed set for reproducibility
- [ ] Statistics reported for each split:
  - Number of examples
  - Task type distribution
  - Average conversation length
  - Token count distribution
- [ ] Splits saved in standardized format (JSON, JSONL, or Hugging Face dataset format)
- [ ] Validation that no data leakage between splits

### Technical Considerations
- Use scikit-learn's `train_test_split` with stratification
- Ensure small classes have sufficient examples in each split
- Consider conversation length balance across splits
- Document split methodology for reproducibility
- Save split indices for reference

### Dependencies
- User Story 1.3 (Data Preprocessing Pipeline)

---

## User Story 1.5: Token Analysis and Max Length Determination

**As a** ML engineer
**I want to** analyze token distributions and determine optimal max_length for training
**So that** I can configure Unsloth training parameters appropriately

### Acceptance Criteria
- [ ] Tokenizer loaded for selected base model (Qwen3-0.6B)
- [ ] Token count statistics computed for all splits (train/val/test):
  - Mean, median, min, max token counts
  - Percentile distribution (25th, 50th, 75th, 90th, 95th, 99th)
- [ ] Recommended max_length determined (based on 95th percentile)
- [ ] Truncation analysis completed:
  - Number of examples that will be truncated at recommended max_length
  - Average tokens over limit for truncated examples
- [ ] Token statistics documented in analysis notebook
- [ ] Edge cases identified (very long conversations)
- [ ] Unit tests verify tokenization utilities work correctly

### Technical Considerations
- Use existing tokenization utilities in `src/data/tokenization.py`
- Balance between coverage (few truncated examples) and efficiency (shorter sequences)
- Qwen3 supports up to 32K tokens, but typical training uses 2K-8K
- Consider memory constraints for batch size planning
- Document which examples will be truncated for manual review

### Dependencies
- User Story 1.4 (Train/Validation/Test Splits)
- Base model selection decision

---

## User Story 1.6: PyTorch Hyperparameter Calculation

**As a** ML engineer
**I want to** calculate optimal PyTorch training hyperparameters
**So that** I can implement a custom training loop with properly tuned settings

### Acceptance Criteria
- [ ] PyTorch hyperparameter calculation:
  - Batch size calculated based on GPU memory constraints
  - Gradient accumulation steps determined for target effective batch size
  - Learning rate, optimizer (AdamW), and scheduler (cosine) configured
  - Memory estimates computed for model, optimizer states, and activations

- [ ] Unit tests verify hyperparameter calculations

### Technical Considerations
- Custom PyTorch training loop will be implemented (not using high-level trainers)
- Calculate optimal batch size based on sequence length and available GPU memory
- Configure PyTorch AdamW optimizer with appropriate betas and weight decay
- Set up cosine annealing scheduler with linear warmup
- Document all calculated hyperparameters for training script implementation

### Dependencies
- User Story 1.5 (Token Analysis)

---

## Phase 1 Definition of Done

All user stories (1.1-1.6) must be completed with:
- [ ] Unit tests implemented and passing
- [ ] Code formatted with `ruff format .`
- [ ] Type hints on all public functions
- [ ] Docstrings added with Args, Returns, Raises
- [ ] No regressions in existing tests
- [ ] README updated with Phase 1 setup instructions
- [ ] Changes committed to git with clear commit messages

## Phase 1 Success Metrics

- Clean, reproducible dataset ready for training (messages format)
- Data preprocessing pipeline handles all 380 examples successfully
- Train/val/test splits are balanced and representative
- Token analysis completed with recommended max_length determined
- Dataset validated for Unsloth SFTTrainer compatibility
- All utilities are tested and documented
- Setup can be reproduced by following README instructions
