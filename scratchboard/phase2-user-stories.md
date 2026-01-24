# Phase 2: Baseline Evaluation - User Stories

## Overview
Phase 2 establishes baseline performance metrics using the untrained base model to quantify the impact of finetuning. This includes zero-shot and few-shot evaluation across quantitative and qualitative dimensions.

---

## User Story 2.1: Base Model Selection and Loading

**As a** ML engineer
**I want to** select and load an appropriate base language model
**So that** I can establish a baseline for finetuning comparison

### Acceptance Criteria
- [ ] Base model selected: Qwen2.5-1.5B-Instruct
- [ ] Model selection rationale documented:
  - Parameter count (< 1B preferred)
  - Memory footprint on T4 16GB
  - Minimal instruction tuning (to demonstrate finetuning impact)
  - Tokenizer compatibility with dataset
- [ ] Model loaded successfully with appropriate dtype (float16/bfloat16)
- [ ] Tokenizer loaded and configured with special tokens
- [ ] Memory profiling completed (model size in VRAM)
- [ ] Model can generate text on sample inputs
- [ ] Model architecture documented (layers, attention heads, hidden size)

### Technical Considerations
- Qwen2.5-1.5B-Instruct selected for modern architecture with tool-use support
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

## User Story 2.3: Zero-Shot Evaluation

**As a** ML engineer
**I want to** evaluate the base model's zero-shot performance on test conversations
**So that** I establish a baseline without any task-specific examples

### Acceptance Criteria
- [ ] All test set examples evaluated with zero-shot prompts
- [ ] Generation parameters documented (temperature=0.7, top_p=0.9, etc.)
- [ ] Generated responses saved with example IDs for analysis
- [ ] Quantitative metrics computed:
  - Perplexity on test conversations
  - ROUGE scores (R-1, R-2, R-L) vs reference answers
  - BLEU score
  - Exact match for binary decisions (in appetite / not in appetite)
- [ ] Results stored in structured format (JSON/CSV)
- [ ] Qualitative sample review completed (10-20 examples)
- [ ] Failure modes identified and categorized

### Technical Considerations
- Use consistent random seed for reproducible generation
- Compute metrics using `evaluate` library from Hugging Face
- Handle cases where model produces invalid/empty responses
- Track generation statistics (tokens generated, time per example)
- Document specific failure patterns (off-topic, lacks domain knowledge, etc.)

### Dependencies
- User Story 2.2 (Inference Pipeline)

---

## User Story 2.4: Few-Shot Evaluation

**As a** ML engineer
**I want to** evaluate the base model with few-shot examples in the prompt
**So that** I can compare finetuning benefits against in-context learning

### Acceptance Criteria
- [ ] Few-shot prompt template implemented (2-3 examples from train set)
- [ ] Examples selected to represent diverse task types
- [ ] Few-shot evaluation completed on test set
- [ ] Quantitative metrics computed (same as zero-shot):
  - Perplexity
  - ROUGE scores
  - BLEU score
  - Exact match for decisions
- [ ] Comparison table created: Zero-shot vs Few-shot
- [ ] Token count analysis for prompt overhead
- [ ] Qualitative improvements documented
- [ ] Few-shot results saved for later comparison

### Technical Considerations
- Select representative few-shot examples (avoid cherry-picking best cases)
- Ensure few-shot prompt fits within context window with test example
- Test multiple few-shot configurations (1, 2, 3 examples)
- Monitor if model copies few-shot examples verbatim
- Validate that improvements are meaningful, not just memorization

### Dependencies
- User Story 2.3 (Zero-Shot Evaluation)

---

## User Story 2.5: Baseline Metrics Dashboard

**As a** ML engineer
**I want to** create a comprehensive baseline metrics dashboard
**So that** I can easily compare finetuned model performance against baselines

### Acceptance Criteria
- [ ] Metrics organized by evaluation type:
  - Zero-shot baseline
  - Few-shot baseline (1, 2, 3 examples)
- [ ] Quantitative metrics table created with:
  - Perplexity
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BLEU score
  - Exact match accuracy
  - Average response length
  - Generation time per example
- [ ] Metrics broken down by task type:
  - Appetite checks
  - Product recommendations
  - Small business eligibility
  - Auto LOB checks
  - General queries
- [ ] Visualizations created (bar charts, distribution plots)
- [ ] Statistical significance tests applied (if applicable)
- [ ] Dashboard exported as HTML/PDF for reference

### Technical Considerations
- Use pandas for data aggregation and analysis
- Create visualizations with matplotlib/seaborn or plotly
- Ensure metrics are reproducible with documented random seeds
- Include confidence intervals where appropriate
- Save raw data for recomputation if needed

### Dependencies
- User Story 2.3 (Zero-Shot Evaluation)
- User Story 2.4 (Few-Shot Evaluation)

---

## User Story 2.6: Qualitative Analysis and Error Categorization

**As a** ML engineer
**I want to** perform qualitative analysis of baseline model outputs
**So that** I identify specific weaknesses that finetuning should address

### Acceptance Criteria
- [ ] Sample of 50-100 test examples manually reviewed
- [ ] Error categories defined and documented:
  - Lack of insurance domain knowledge
  - Poor multi-turn coherence
  - Incorrect risk assessment
  - Off-topic or irrelevant responses
  - Failure to reference company profile
  - Generic/unhelpful answers
- [ ] Error frequency distribution computed
- [ ] Representative examples selected for each error type
- [ ] Qualitative findings documented in markdown report
- [ ] Specific improvement targets identified for finetuning
- [ ] Edge cases and challenging examples flagged for attention

### Technical Considerations
- Use structured review protocol for consistency
- Include both zero-shot and few-shot outputs in review
- Note positive behaviors to preserve during finetuning
- Document whether errors are due to:
  - Lack of knowledge (addressable via finetuning)
  - Inability to follow instructions (needs better prompting)
  - Model capacity limitations (may not be fixable)
- Create annotated examples for later comparison

### Dependencies
- User Story 2.3 (Zero-Shot Evaluation)
- User Story 2.4 (Few-Shot Evaluation)

---

## User Story 2.7: Baseline Report Generation

**As a** ML engineer
**I want to** create a comprehensive baseline evaluation report
**So that** stakeholders understand pre-finetuning model performance and limitations

### Acceptance Criteria
- [ ] Report document created (markdown or notebook format) with:
  - Executive summary of baseline performance
  - Model selection rationale
  - Evaluation methodology
  - Quantitative results with tables and charts
  - Qualitative analysis with example outputs
  - Error analysis and categorization
  - Key findings and insights
  - Recommendations for finetuning approach
- [ ] Report includes side-by-side comparisons of:
  - Base model outputs
  - Reference answers
  - Error annotations
- [ ] Report is self-contained and readable by non-technical stakeholders
- [ ] All visualizations are publication-quality
- [ ] Report saved in project documentation

### Technical Considerations
- Use Jupyter notebook or Quarto for reproducible report generation
- Include enough detail for reproducibility
- Balance technical depth with accessibility
- Highlight specific examples that demonstrate need for finetuning
- Version control the report alongside code

### Dependencies
- User Story 2.5 (Baseline Metrics Dashboard)
- User Story 2.6 (Qualitative Analysis)

---

## Phase 2 Definition of Done

All user stories (2.1-2.7) must be completed with:
- [ ] Unit tests implemented and passing
- [ ] Code formatted with `ruff format .`
- [ ] Type hints on all public functions
- [ ] Docstrings added with Args, Returns, Raises
- [ ] No regressions in existing tests
- [ ] README updated with baseline evaluation results
- [ ] Changes committed to git with clear commit messages

## Phase 2 Success Metrics

- Base model loaded and generating outputs successfully
- Zero-shot and few-shot baselines established with quantitative metrics
- Clear documentation of baseline performance gaps
- Error categories identified to guide finetuning priorities
- Comprehensive baseline report completed
- All evaluation code is reusable for finetuned model comparison
