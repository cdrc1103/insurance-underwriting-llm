# Phase 4: Evaluation & Analysis - User Stories

## Overview
Phase 4 evaluates the finetuned model on the held-out test set, compares performance against baselines, and performs comprehensive analysis to understand improvements, limitations, and failure modes.

---

## User Story 4.1: Finetuned Model Loading and Validation

**As a** ML engineer
**I want to** load the finetuned model with LoRA adapters and validate setup
**So that** I can perform evaluation with confidence in model integrity

### Acceptance Criteria
- [ ] Base model loaded successfully
- [ ] LoRA adapters loaded and merged with base model
- [ ] Model device placement verified (GPU if available)
- [ ] Tokenizer loaded with correct configuration
- [ ] Smoke test generation completed on sample inputs
- [ ] Generated outputs are coherent and domain-relevant
- [ ] Memory footprint validated (fits in GPU memory)
- [ ] Model versioning information logged (checkpoint ID, training date)
- [ ] Adapter merge verification (parameters updated correctly)

### Technical Considerations
- Use `peft.PeftModel.from_pretrained()` for adapter loading
- Consider using merged model vs adapter-only for inference speed
- Verify model is in eval mode (`model.eval()`)
- Test generation with same parameters as baseline for fair comparison
- Document any inference optimizations (quantization, torch.compile)

### Dependencies
- Phase 3 completed (trained model available)
- Phase 2 inference pipeline reusable

---

## User Story 4.2: Test Set Evaluation with Quantitative Metrics

**As a** ML engineer
**I want to** evaluate the finetuned model on the test set with comprehensive metrics
**So that** I can quantify performance improvements over baseline

### Acceptance Criteria
- [ ] All test set examples evaluated with finetuned model
- [ ] Generation parameters matched to baseline (temperature, top_p, etc.)
- [ ] Quantitative metrics computed:
  - Perplexity on test conversations
  - ROUGE scores (R-1, R-2, R-L) vs reference answers
  - BLEU score
  - Exact match for binary decisions
  - F1 score for classification tasks
- [ ] Metrics broken down by task type:
  - Appetite checks
  - Product recommendations
  - Small business eligibility
  - Auto LOB checks
  - General queries
- [ ] Statistical significance testing performed (if applicable)
- [ ] Generated responses saved for qualitative analysis
- [ ] Evaluation results exported to structured format (JSON/CSV)

### Technical Considerations
- Reuse evaluation pipeline from Phase 2 for consistency
- Use same test set (no leakage from training/validation)
- Compute confidence intervals for metrics
- Consider additional metrics: BERTScore, semantic similarity
- Track generation time and compare to baseline
- Handle edge cases (empty generations, errors)

### Dependencies
- User Story 4.1 (Model Loading)
- Phase 2 baseline metrics available for comparison

---

## User Story 4.3: Baseline vs Finetuned Comparison Analysis

**As a** ML engineer
**I want to** create comprehensive comparison between baseline and finetuned models
**So that** I can demonstrate the impact of finetuning quantitatively

### Acceptance Criteria
- [ ] Comparison table created with all metrics:
  - Zero-shot baseline
  - Few-shot baseline
  - Finetuned model
- [ ] Percentage improvement calculated for each metric
- [ ] Statistical significance tests applied (t-test, paired tests)
- [ ] Comparison visualizations created:
  - Bar charts for metric comparisons
  - Radar charts for multi-metric view
  - Task-type breakdown comparisons
- [ ] Win/tie/loss analysis: count of examples where finetuned > baseline
- [ ] Performance summary with key findings:
  - Biggest improvements
  - Areas still needing work
  - Unexpected results
- [ ] Comparison report exported (HTML/PDF)

### Technical Considerations
- Use consistent evaluation settings across all models
- Consider multiple comparison dimensions (overall, by task, by difficulty)
- Apply multiple comparison corrections for statistical tests
- Highlight both relative and absolute improvements
- Document any confounding factors
- Include sample size in statistical tests

### Dependencies
- User Story 4.2 (Test Set Evaluation)
- Phase 2 baseline results

---

## User Story 4.4: Qualitative Analysis of Model Outputs

**As a** ML engineer
**I want to** perform detailed qualitative analysis of finetuned model outputs
**So that** I understand what the model learned and identify remaining weaknesses

### Acceptance Criteria
- [ ] Sample of 50-100 test examples manually reviewed
- [ ] Side-by-side comparison created:
  - Baseline model output
  - Finetuned model output
  - Reference answer
- [ ] Improvements categorized:
  - Better insurance domain knowledge
  - Improved multi-turn coherence
  - More accurate risk assessments
  - Better use of company context
  - More professional tone
- [ ] Remaining errors categorized:
  - Persistent knowledge gaps
  - Edge case failures
  - Hallucinations or incorrect facts
  - Formatting issues
- [ ] Example outputs selected for showcase:
  - Best improvements
  - Interesting edge cases
  - Failure modes
- [ ] Qualitative findings documented in analysis report

### Technical Considerations
- Use structured review protocol for consistency
- Include diverse examples across task types
- Note both improvements and regressions
- Consider blind review (reviewer doesn't know which is finetuned)
- Create annotated examples with highlighted improvements
- Document specific insurance concepts the model learned

### Dependencies
- User Story 4.2 (Test Set Evaluation)
- Phase 2 baseline outputs for comparison

---

## User Story 4.5: Error Analysis and Failure Mode Identification

**As a** ML engineer
**I want to** conduct systematic error analysis on finetuned model failures
**So that** I identify remaining limitations and guide future improvements

### Acceptance Criteria
- [ ] Failure cases identified (examples where model performs poorly)
- [ ] Error taxonomy created with categories:
  - Knowledge errors (incorrect facts, missing domain knowledge)
  - Reasoning errors (wrong conclusions from correct info)
  - Context errors (not using company profile properly)
  - Formatting errors (poor structure, incomplete responses)
  - Consistency errors (contradictions within response)
- [ ] Error frequency distribution computed for each category
- [ ] Root cause analysis performed for common errors:
  - Insufficient training data for specific scenarios
  - Model capacity limitations
  - Bias in training distribution
  - Tokenization or formatting issues
- [ ] Challenging examples identified (where model struggles most)
- [ ] Recommendations documented for future iterations:
  - Data augmentation needs
  - Additional training strategies
  - Prompt engineering improvements
- [ ] Error analysis report created with annotated examples

### Technical Considerations
- Focus on actionable insights for improvement
- Compare error types to baseline error analysis (from Phase 2)
- Consider whether errors are acceptable given model size
- Identify if errors cluster around specific task types or companies
- Document whether errors pose risks for production use
- Consider inter-annotator agreement for subjective judgments

### Dependencies
- User Story 4.2 (Test Set Evaluation)
- User Story 4.4 (Qualitative Analysis)

---

## User Story 4.6: Multi-Turn Conversation Quality Assessment

**As a** ML engineer
**I want to** assess how well the model maintains context across conversation turns
**So that** I verify the primary objective of multi-turn coherence is achieved

### Acceptance Criteria
- [ ] Multi-turn coherence metrics defined:
  - Reference tracking (pronouns, company mentions)
  - Consistency across turns (no contradictions)
  - Context accumulation (using previous turns)
  - Conversation flow (natural transitions)
- [ ] Quantitative coherence analysis:
  - Perplexity per turn (does it degrade?)
  - Entity tracking accuracy
  - Response relevance to conversation history
- [ ] Qualitative coherence review:
  - Long conversation examples (5+ turns)
  - Identification of context loss or drift
  - Comparison to baseline multi-turn performance
- [ ] Attention analysis (optional):
  - Which conversation turns model attends to
  - Company profile attention throughout conversation
- [ ] Multi-turn assessment report with examples
- [ ] Coherence improvement quantified vs baseline

### Technical Considerations
- Create or use existing multi-turn metrics (rare in literature)
- Consider using coreference resolution for entity tracking
- Test with conversations of varying lengths
- Check if model maintains facts stated in earlier turns
- Document at what turn count (if any) coherence breaks down
- Compare to human-written conversations for context

### Dependencies
- User Story 4.2 (Test Set Evaluation)
- User Story 4.4 (Qualitative Analysis)

---

## User Story 4.7: Insurance Domain Knowledge Assessment

**As a** ML engineer
**I want to** assess the model's acquisition of insurance domain knowledge
**So that** I verify the model learned insurance concepts beyond generic responses

### Acceptance Criteria
- [ ] Domain-specific terms and concepts identified in model outputs:
  - Insurance product types (GL, WC, Property, Auto)
  - Underwriting criteria (revenue limits, employee counts)
  - Risk factors (industry type, location, business operations)
  - Appetite guidelines (in/out of appetite)
- [ ] Domain knowledge test cases created:
  - Questions requiring specific insurance knowledge
  - Edge cases with uncommon industries or situations
  - Scenarios requiring guideline interpretation
- [ ] Comparison of domain language usage:
  - Baseline model (generic or incorrect terms)
  - Finetuned model (appropriate domain terminology)
  - Reference answers
- [ ] Domain knowledge accuracy assessed:
  - Correct application of revenue/employee limits
  - Accurate product recommendations
  - Proper risk factor identification
- [ ] Knowledge gaps documented (concepts not learned)
- [ ] Domain learning report with annotated examples

### Technical Considerations
- Create checklist of key insurance concepts to assess
- Verify factual accuracy of insurance recommendations
- Check if model generalizes beyond training examples
- Identify if model memorized guidelines vs learned principles
- Document novel applications of domain knowledge
- Consider expert review of insurance-specific outputs

### Dependencies
- User Story 4.2 (Test Set Evaluation)
- User Story 4.4 (Qualitative Analysis)
- Domain expert consultation (if available)

---

## User Story 4.8: Performance Report and Insights Documentation

**As a** ML engineer
**I want to** create a comprehensive performance report with insights and recommendations
**So that** stakeholders understand finetuning results and next steps

### Acceptance Criteria
- [ ] Performance report document created with sections:
  - Executive summary (key results, improvements)
  - Methodology (model, training, evaluation approach)
  - Quantitative results (tables, charts, statistical tests)
  - Qualitative findings (examples, improvements, limitations)
  - Error analysis (failure modes, root causes)
  - Multi-turn coherence assessment
  - Domain knowledge evaluation
  - Comparison to baseline
  - Key insights and learnings
  - Recommendations for production or future work
- [ ] Report includes:
  - Publication-quality visualizations
  - Annotated example conversations
  - Side-by-side comparisons
  - Statistical significance indicators
- [ ] Success criteria evaluation (from Phase 1):
  - Multi-turn context maintenance: ✓/✗
  - Accurate recommendations: ✓/✗
  - Insurance terminology: ✓/✗
  - Measurable improvement: ✓/✗
- [ ] Report is self-contained and readable
- [ ] Report exported as PDF/HTML
- [ ] Raw data and artifacts archived for reproducibility

### Technical Considerations
- Use Jupyter notebook or Quarto for reproducible report
- Balance technical depth with accessibility
- Include enough detail for peer review
- Highlight both successes and limitations honestly
- Provide context for results (model size, training data, etc.)
- Include appendix with full metric tables

### Dependencies
- All other Phase 4 user stories completed
- Phase 3 training results documented

---

## User Story 4.9: Model Demonstration Notebook

**As a** ML engineer
**I want to** create an interactive demonstration notebook
**So that** users can easily interact with the finetuned model and see its capabilities

### Acceptance Criteria
- [ ] Jupyter notebook created with sections:
  - Setup and model loading
  - Example company profiles
  - Interactive conversation interface
  - Pre-generated example conversations
  - Comparison to baseline (side-by-side)
  - Performance metrics summary
  - Usage instructions
- [ ] Notebook includes 5-10 diverse example conversations
- [ ] Interactive input widgets for custom queries (optional)
- [ ] Notebook runs end-to-end without errors
- [ ] Clear instructions for running locally or on Colab
- [ ] Notebook includes:
  - Model information and training details
  - Inference parameters used
  - Limitations and disclaimers
- [ ] Notebook is well-documented with markdown cells
- [ ] Exported HTML version for easy sharing

### Technical Considerations
- Keep dependencies minimal for easy setup
- Include fallback for CPU-only execution
- Add estimated runtime warnings for generation
- Consider hosting notebook on Colab or Hugging Face Spaces
- Include clear examples of good and mediocre outputs
- Add troubleshooting section for common issues

### Dependencies
- User Story 4.1 (Model Loading)
- User Story 4.8 (Performance Report for examples)

---

## Phase 4 Definition of Done

All user stories (4.1-4.9) must be completed with:
- [ ] Unit tests implemented and passing
- [ ] Code formatted with `ruff format .`
- [ ] Type hints on all public functions
- [ ] Docstrings added with Args, Returns, Raises
- [ ] No regressions in existing tests
- [ ] README updated with evaluation results summary
- [ ] Changes committed to git with clear commit messages

## Phase 4 Success Metrics

- Finetuned model shows measurable improvement over baseline
- Quantitative metrics demonstrate statistical significance where applicable
- Multi-turn coherence is improved compared to baseline
- Model demonstrates acquisition of insurance domain knowledge
- Comprehensive performance report documents results
- Interactive demo notebook allows easy model exploration
- Error analysis provides actionable insights for future work
- All evaluation artifacts are saved and reproducible
