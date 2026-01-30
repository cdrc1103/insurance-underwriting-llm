# Phase 2: Baseline Evaluation - User Stories

## Overview
Phase 2 establishes baseline performance metrics using the untrained base model to quantify the impact of finetuning. Evaluation uses **Model-as-Judge** (Claude) to assess domain-specific accuracy on appetite decisions, product recommendations, risk assessment, and limit/deductible suggestions.

---

## User Story 2.1: Base Model Selection and Loading

**As a** ML engineer
**I want to** select and load an appropriate base language model
**So that** I can establish a baseline for finetuning comparison

### Acceptance Criteria
- [ ] Base model selected: Qwen3-0.6B
- [ ] Model loaded successfully with appropriate dtype (float16/bfloat16)
- [ ] Tokenizer loaded and configured with special tokens
- [ ] Memory profiling completed (model size in VRAM)
- [ ] Model can generate text on sample inputs
- [ ] Model architecture documented (layers, attention heads, hidden size)

### Technical Considerations
- Qwen3-0.6B selected for modern architecture with tool-use support
- Test generation quality on sample conversations
- Verify model fits in 16GB VRAM with headroom for inference
- Check tokenizer vocabulary size and coverage of insurance terms
- Consider using 4-bit/8-bit quantization for larger models

### Dependencies
- Phase 1 completed (data ready for evaluation)

---

## User Story 2.2: Inference Pipeline Implementation

**As a** ML engineer
**I want to** implement a reusable inference pipeline for generating responses
**So that** I can evaluate the model on test conversations consistently

### Acceptance Criteria
- [ ] Inference function implemented with configurable parameters:
  - Temperature, top_p, top_k
  - Max generation length
  - Repetition penalty
  - Stopping criteria
- [ ] Company profile + conversation history formatted as input prompt
- [ ] Generation handles multi-turn context correctly
- [ ] Response extraction from model output implemented
- [ ] Batched inference supported for efficiency
- [ ] Generation parameters logged for reproducibility
- [ ] Error handling for generation failures
- [ ] Unit tests verify prompt formatting and response extraction

### Technical Considerations
- Use `model.generate()` with appropriate sampling strategy
- Implement proper prompt template matching model's expected format
- Handle tokenizer padding and truncation
- Consider caching key-value states for multi-turn efficiency
- Monitor generation time per example
- Validate outputs don't contain hallucinated company information

### Dependencies
- User Story 2.1 (Base Model Loading)

---

## User Story 2.3: Zero-Shot Evaluation with Model-as-Judge

**As a** ML engineer
**I want to** evaluate the base model's zero-shot performance using Claude as a judge
**So that** I establish a domain-accurate baseline for insurance underwriting tasks

### Acceptance Criteria
- [ ] All test set examples (or sample of 50-100) evaluated with zero-shot prompts
- [ ] Generation parameters documented (temperature=0.7, top_p=0.9, etc.)
- [ ] Generated responses saved with example IDs for analysis
- [ ] Model-as-Judge evaluation system implemented:
  - Claude API integration for scoring responses
  - Evaluation rubric designed for insurance domain:
    * Appetite decision accuracy (in/out/qualified)
    * Product recommendation relevance
    * Limit/deductible accuracy
    * Risk assessment correctness
    * Use of company profile information
    * Multi-turn coherence
  - Scoring scale (1-5 or binary correct/incorrect per criteria)
- [ ] Evaluation results stored in structured format (JSON/CSV)
- [ ] Aggregate metrics computed:
  - Overall accuracy by task type
  - Per-criteria performance scores
  - Common error patterns identified
- [ ] Sample outputs manually reviewed to validate judge accuracy

### Technical Considerations
- Use Claude API (Anthropic SDK) for evaluation
- Design clear, objective rubric to minimize judge bias
- Include reference answer in judge prompt for comparison
- Batch API calls for cost efficiency
- Handle API rate limits and retries
- Validate that judge outputs match manual spot-checks
- Consider temperature=0 for judge to ensure consistency

### Dependencies
- User Story 2.2 (Inference Pipeline)

---

## User Story 2.4: Baseline Results Analysis and Documentation

**As a** ML engineer
**I want to** analyze and document baseline evaluation results in a notebook
**So that** I can identify specific weaknesses and track finetuning improvements

### Acceptance Criteria
- [ ] Evaluation results analyzed in Jupyter notebook:
  - Overall accuracy by task type
  - Per-criteria performance breakdown
  - Error pattern analysis from judge outputs
  - Sample outputs with annotations
- [ ] Key findings documented:
  - Most common failure modes
  - Strengths to preserve during finetuning
  - Specific improvement targets
- [ ] Comparison framework established for post-finetuning:
  - Baseline scores saved for reference
  - Evaluation rubric documented for reuse
  - Same test examples marked for re-evaluation
- [ ] Notebook includes visualizations:
  - Performance by task type (bar charts)
  - Error distribution (pie/bar charts)
  - Sample outputs (side-by-side with references)

### Technical Considerations
- Use pandas for data aggregation
- Create clear, reproducible visualizations
- Document judge prompt and rubric for consistency
- Save baseline results as JSON/CSV for comparison
- Include enough examples to demonstrate failure patterns

### Dependencies
- User Story 2.3 (Zero-Shot Evaluation)

---

## Phase 2 Definition of Done

All user stories (2.1-2.4) must be completed with:
- [ ] Unit tests implemented and passing
- [ ] Code formatted with `ruff format .`
- [ ] Type hints on all public functions
- [ ] Docstrings added with Args, Returns, Raises
- [ ] No regressions in existing tests
- [ ] README updated with baseline evaluation approach
- [ ] Changes committed to git with clear commit messages

## Phase 2 Success Metrics

- Base model loaded and generating outputs successfully
- Zero-shot baseline established using Model-as-Judge (Claude)
- Domain-specific accuracy metrics computed (appetite, products, limits, risk)
- Clear documentation of baseline performance gaps
- Error patterns identified to guide finetuning priorities
- Evaluation framework is reusable for finetuned model comparison
- All evaluation runs in notebooks for transparency and reproducibility
