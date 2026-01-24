# Insurance Underwriting LLM - Project Scope

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
Finetune Qwen2.5-1.5B-Instruct to handle multi-turn insurance underwriting conversations with tool-use capabilities using the Snorkel AI Multi-Turn Insurance Underwriting dataset.

### Success Criteria
1. Model can maintain context across multiple conversation turns
2. Model generates accurate underwriting recommendations based on company information
3. Model demonstrates understanding of insurance terminology and concepts
4. Measurable improvement over base model on evaluation metrics

## Dataset

**Source:** [snorkelai/Multi-Turn-Insurance-Underwriting](https://huggingface.co/datasets/snorkelai/Multi-Turn-Insurance-Underwriting)

**Size:** 380 multi-turn conversation examples

**Task Types:**
- Appetite checks (eligibility assessment)
- Product recommendations
- Small business eligibility
- Auto LOB appetite checks
- General underwriting queries

**Data Structure:**
- Company details: revenue, employees, payroll, vehicles, location, business description
- Multi-turn conversation traces with underwriter and assistant roles
- Reference answers for evaluation
- Correctness labels

## Approach

### Model Selection

**Selected Model:** Qwen/Qwen2.5-1.5B-Instruct

**Why Qwen2.5-1.5B-Instruct?**
- Modern architecture (RoPE, GQA, 128K context support)
- Native function calling / tool-use support
- Structured output (JSON, XML) generation
- Trained on 18 trillion tokens
- Apache 2.0 license
- Fits on T4 16GB with QLoRA

**Why NOT older architectures (GPT-2, OPT, Pythia)?**
- No RoPE embeddings → worse position/context handling for multi-turn
- Older attention mechanisms → less efficient learning
- Training recipes from 2019-2022 are inferior to modern techniques
- 380 examples can't teach both domain knowledge AND conversational coherence
- Fine-tuning can't fix fundamental architectural limitations

**Rationale:** Using an instruction-tuned model for domain adaptation is what companies actually do in production. The project demonstrates teaching domain-specific tool usage, not basic conversation.

### Finetuning Strategy

**LoRA/QLoRA:** Parameter-efficient finetuning to:
- Reduce memory requirements (fit on free GPU)
- Minimize overfitting risk with 380 examples
- Enable quick experimentation

**LoRA Configuration:**
- Rank (r): 8-16 (to be tuned)
- Alpha: 16-32
- Target modules: Query/Key/Value attention matrices
- Dropout: 0.05-0.1

**QLoRA Benefits:**
- 4-bit quantization of base model
- LoRA adapters in float16/bfloat16
- Further memory reduction for larger models

### Training Setup

**Data Preparation:**
- Extract company information and conversation turns
- Format as multi-turn dialogue (underwriter question → assistant response)
- **Include tool calls** - train for domain-specific tool use
- Create train/validation/test splits (e.g., 300/40/40)

**Input Format:**
```
Company Profile:
- Name: [company_name]
- Revenue: [annual_revenue]
- Employees: [number_of_employees]
- Industry: [industry from description]
- State: [state]

Conversation:
User: [underwriter question]
Assistant: [model response]
User: [follow-up question]
Assistant: [model response]
...
```

**Training Hyperparameters (initial):**
- Learning rate: 1e-4 to 3e-4
- Batch size: 4-8 (with gradient accumulation)
- Epochs: 3-5
- Max sequence length: 1024-2048 tokens
- Optimizer: AdamW with warmup
- Scheduler: Cosine with warmup

**Hardware Target:**
- Lightning AI Free Tier: T4 16GB VRAM

## Scope Boundaries

### In Scope
✅ Multi-turn dialogue generation
✅ Tool calling for underwriting operations
✅ Company profile understanding
✅ Insurance domain knowledge adaptation
✅ Underwriting reasoning and recommendations
✅ LoRA/QLoRA parameter-efficient finetuning
✅ Evaluation on held-out test set
✅ Comparison of base model vs finetuned model

### Out of Scope
❌ External API integrations (mock tools only)
❌ Retrieval-augmented generation (RAG)
❌ Full finetuning (only LoRA/QLoRA)
❌ Multi-agent systems
❌ Production deployment
❌ Real-time inference optimization
❌ Model compression beyond quantization

## Evaluation Metrics

### Quantitative
1. **Perplexity** on validation/test conversations
2. **ROUGE scores** (R-1, R-2, R-L) comparing generated vs reference responses
3. **BLEU score** for response quality
4. **Exact match** for binary decisions (in appetite / not in appetite)
5. **F1 score** for classification tasks

### Qualitative
1. Conversational coherence across turns
2. Factual accuracy of recommendations
3. Domain terminology usage
4. Reasoning quality
5. Response relevance to company context

### Comparison
- Base model (zero-shot) vs Finetuned model
- Base model (few-shot with examples) vs Finetuned model

## Implementation Phases

### Phase 1: Environment & Data Setup
- Set up development environment
- Download and explore dataset
- Implement data loading and preprocessing
- Create train/val/test splits
- Format data for multi-turn dialogue

### Phase 2: Baseline Evaluation
- Select base model
- Implement inference pipeline
- Evaluate base model (zero-shot and few-shot)
- Establish baseline metrics

### Phase 3: Model Finetuning
- Implement LoRA/QLoRA configuration
- Set up training loop with logging
- Train model on prepared dataset
- Monitor training metrics and convergence

### Phase 4: Evaluation & Analysis
- Evaluate finetuned model on test set
- Compare against baseline
- Generate example conversations
- Analyze failure cases
- Document findings

### Phase 5: Documentation & Showcase
- Clean up code and notebooks
- Write comprehensive README
- Create example inference notebook
- Document results and insights
- Prepare for portfolio/GitHub

## Technical Stack

**Core Libraries:**
- `transformers` (Hugging Face) - Model loading and training
- `peft` (Parameter-Efficient Fine-Tuning) - LoRA/QLoRA implementation
- `datasets` (Hugging Face) - Dataset handling
- `bitsandbytes` - 4-bit quantization for QLoRA
- `accelerate` - Distributed training utilities

**Training & Evaluation:**
- `torch` - PyTorch backend
- `wandb` or `tensorboard` - Training monitoring
- `evaluate` - Metrics computation
- `rouge_score`, `sacrebleu` - Text generation metrics

**Development:**
- Python 3.9+
- Python files for experimentation and final implementation
- Git for version control

## Deliverables

1. **Cleaned dataset** in processed format ready for training
2. **Training scripts** for reproducible finetuning
3. **Finetuned model** with LoRA adapters (pushed to Hugging Face Hub)
4. **Evaluation results** comparing base vs finetuned performance
5. **Demo notebook** showing inference on new examples
6. **Documentation** explaining approach, results, and usage
7. **README** with clear setup instructions and results summary

## Success Indicators

- Model generates coherent multi-turn conversations about insurance underwriting
- Quantitative metrics show clear improvement over baseline
- Model demonstrates insurance domain knowledge not present in base model
- Entire training pipeline runs on free-tier GPU
- Code is clean, documented, and reproducible

## Timeline Estimate

- Phase 1: Data Setup - 2-3 hours
- Phase 2: Baseline - 1-2 hours
- Phase 3: Finetuning - 3-4 hours (including experimentation)
- Phase 4: Evaluation - 2-3 hours
- Phase 5: Documentation - 2-3 hours

**Total:** ~10-15 hours of focused work

## Learning Objectives

This project demonstrates:
1. Practical experience with LoRA/QLoRA finetuning
2. Working with domain-specific datasets
3. Multi-turn dialogue system development
4. Evaluation methodology for text generation
5. Resource-constrained model training
6. End-to-end ML project execution

---

**Note:** This is a showcase/learning project. The focus is on demonstrating finetuning skills and understanding of LLM adaptation. However, the ambition of the implementation should be to use this in production.
