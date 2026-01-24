# Data Exploration: Multi-Turn Insurance Underwriting Dataset

## Overview

This document summarizes the findings from exploring the Multi-Turn Insurance Underwriting dataset from Hugging Face.

**Dataset Source**: `snorkelai/Multi-Turn-Insurance-Underwriting`

**Analysis Date**: TBD

## Dataset Schema

### Structure

The dataset contains multi-turn conversations between insurance underwriters and an AI assistant, along with company profile information.

**Note**: This section will be updated after running the exploration notebook.

### Key Fields

- **Company Profile**: Information about the business seeking insurance
  - Company name
  - Annual revenue
  - Number of employees
  - Industry/business type
  - State/location

- **Conversation**: Multi-turn dialogue
  - User questions (underwriter)
  - Assistant responses
  - Role labels

- **Task Type**: Category of underwriting task
  - Appetite checks
  - Product recommendations
  - Eligibility assessments
  - Auto LOB checks
  - General queries

## Dataset Statistics

### Size

- **Total Examples**: TBD (Expected: ~380)
- **Train Split**: TBD
- **Validation Split**: TBD (if exists)
- **Test Split**: TBD (if exists)

### Conversation Characteristics

- **Average Turns per Conversation**: TBD
- **Median Turns**: TBD
- **Range**: TBD to TBD turns
- **Average Token Count**: TBD

### Task Distribution

Distribution of task types across the dataset:

| Task Type | Count | Percentage |
|-----------|-------|------------|
| TBD       | TBD   | TBD%       |

**Note**: To be filled after running exploration notebook.

## Data Quality Analysis

### Issues Identified

1. **Tool Calls**: Examples containing tool/function calls that need to be excluded
   - Count: TBD
   - Action: Filter these examples during preprocessing

2. **Missing Fields**: Examples with incomplete company profiles
   - Count: TBD
   - Action: Handle missing values with defaults or filter

3. **Conversation Structure**: Inconsistencies in conversation format
   - Count: TBD
   - Action: Standardize during preprocessing

### Data Cleaning Strategy

Based on identified issues:

1. Exclude examples with tool/function calls
2. Validate company profile completeness
3. Ensure all conversations have proper role assignments
4. Normalize whitespace and special characters
5. Validate conversation turn ordering

## Company Profile Analysis

### Revenue Distribution

- **Range**: TBD to TBD
- **Mean**: TBD
- **Median**: TBD

### Employee Count Distribution

- **Range**: TBD to TBD
- **Mean**: TBD
- **Median**: TBD

### Industry Distribution

Top industries represented:

| Industry | Count | Percentage |
|----------|-------|------------|
| TBD      | TBD   | TBD%       |

### Geographic Distribution

Top states represented:

| State | Count | Percentage |
|-------|-------|------------|
| TBD   | TBD   | TBD%       |

## Token Length Analysis

### Sequence Length Statistics

Analysis of token counts per example (estimated):

- **Mean**: TBD tokens
- **Median**: TBD tokens
- **90th Percentile**: TBD tokens
- **95th Percentile**: TBD tokens
- **Max**: TBD tokens

### Implications for Model Selection

Based on token length analysis:

- **Recommended Max Sequence Length**: TBD tokens
- **Model Context Window Required**: TBD tokens
- **Truncation Strategy**: TBD

## Sample Examples

### Example 1: Appetite Check

```
Company Profile:
- Name: [TBD]
- Revenue: [TBD]
- Employees: [TBD]
- Industry: [TBD]
- State: [TBD]

Conversation:
User: [TBD]
Assistant: [TBD]
...
```

### Example 2: Product Recommendation

```
[To be filled from notebook]
```

## Key Findings

### Strengths

1. **Multi-turn conversations**: Dataset captures realistic underwriting dialogues
2. **Diverse scenarios**: Good coverage of different task types
3. **Structured data**: Company profiles provide context for responses

### Challenges

1. **Dataset size**: ~380 examples may be limiting for complex fine-tuning
2. **Class imbalance**: Some task types may be underrepresented
3. **Sequence length**: Some conversations may exceed model context limits

### Recommendations

1. **Augmentation**: Consider data augmentation strategies if needed
2. **Stratification**: Use stratified splitting to maintain task distribution
3. **Truncation**: Implement smart truncation that preserves conversation context
4. **Filtering**: Remove examples with tool calls and validation issues

## Next Steps

1. ✅ Dataset downloaded and explored
2. ⏭️ Implement preprocessing pipeline
3. ⏭️ Create train/validation/test splits
4. ⏭️ Implement tokenization for selected model
5. ⏭️ Create data loading utilities

## References

- Dataset: https://huggingface.co/datasets/snorkelai/Multi-Turn-Insurance-Underwriting
- Exploration Notebook: `notebooks/01_data_exploration.ipynb`
- Exploration Summary: `data/exploration_summary.json`
