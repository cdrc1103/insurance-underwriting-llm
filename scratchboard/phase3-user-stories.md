# Phase 3: Model Finetuning - User Stories

## Overview
Phase 3 implements parameter-efficient finetuning using LoRA/QLoRA to adapt the base model for insurance underwriting conversations. This phase includes configuration, training loop implementation, monitoring, and convergence analysis.

---

## User Story 3.1: LoRA/QLoRA Configuration

**As a** ML engineer
**I want to** configure LoRA or QLoRA for parameter-efficient finetuning
**So that** I can train the model within GPU memory constraints (T4 16GB)

### Acceptance Criteria
- [ ] LoRA/QLoRA decision made based on model size and memory profiling
- [ ] PEFT configuration implemented with parameters:
  - Rank (r): 8-16
  - Alpha: 16-32
  - Target modules: Query/Key/Value projection layers
  - Dropout: 0.05-0.1
  - Task type: Causal language modeling
- [ ] For QLoRA: 4-bit quantization configured via bitsandbytes
- [ ] Base model frozen, only LoRA adapters trainable
- [ ] Trainable parameter count reported (should be < 1% of base model)
- [ ] Memory footprint validated (model + optimizer + gradients fit in 16GB)
- [ ] Configuration saved for reproducibility
- [ ] Unit tests verify adapter injection

### Technical Considerations
- Use `peft.LoraConfig` or `peft.BnbConfig` for configuration
- Test multiple rank values (8, 16) to balance capacity vs efficiency
- Consider targeting additional modules (dense, layer norms) if memory allows
- For QLoRA, use NF4 quantization with double quantization
- Verify gradient checkpointing is enabled for memory savings
- Document tradeoffs between LoRA rank and model performance

### Dependencies
- Phase 2 completed (baseline established)
- Base model selected and loaded

---

## User Story 3.2: Training Configuration and Hyperparameters

**As a** ML engineer
**I want to** configure training hyperparameters and setup
**So that** the model trains effectively without overfitting on 300 training examples

### Acceptance Criteria
- [ ] Training configuration defined:
  - Learning rate: 1e-4 to 3e-4 (with experimentation)
  - Batch size: 4-8 per GPU
  - Gradient accumulation steps: Calculate for effective batch size of 16-32
  - Epochs: 3-5
  - Max sequence length: 1024-2048 tokens
  - Warmup steps: 10% of total steps
  - Weight decay: 0.01
- [ ] Optimizer configured (AdamW with betas)
- [ ] Learning rate scheduler configured (cosine with warmup)
- [ ] Gradient clipping enabled (max_grad_norm=1.0)
- [ ] Mixed precision training configured (fp16 or bf16)
- [ ] Random seeds set for reproducibility (torch, numpy, random)
- [ ] Configuration exported to file (JSON/YAML)

### Technical Considerations
- Use `transformers.TrainingArguments` for standardized configuration
- Calculate total training steps: (examples * epochs) / (batch_size * gradient_accumulation * num_gpus)
- Consider learning rate warmup to stabilize early training
- Monitor gradient norms to detect vanishing/exploding gradients
- Test with smaller learning rates to avoid catastrophic forgetting
- Enable logging at reasonable intervals (every 10-50 steps)

### Dependencies
- User Story 3.1 (LoRA Configuration)
- Phase 1 data loading utilities

---

## User Story 3.3: Training Loop with Logging

**As a** ML engineer
**I want to** implement a training loop with comprehensive logging
**So that** I can monitor training progress and diagnose issues in real-time

### Acceptance Criteria
- [ ] Training loop implemented using `transformers.Trainer` or custom loop
- [ ] Logging configured for:
  - Training loss (per step and epoch)
  - Learning rate (per step)
  - Gradient norm
  - Validation loss (per epoch)
  - Memory usage (GPU utilization)
  - Training time per epoch
- [ ] Integration with experiment tracking (Weights & Biases or TensorBoard)
- [ ] Checkpointing implemented:
  - Save best model based on validation loss
  - Save intermediate checkpoints every N steps
  - Save optimizer state for resumption
- [ ] Early stopping configured based on validation loss
- [ ] Progress bars for user-friendly monitoring
- [ ] Logging output saved to file for post-training analysis

### Technical Considerations
- Use `transformers.TrainerCallback` for custom logging
- Log to both console and experiment tracker
- Implement checkpoint cleanup to save disk space
- Monitor for signs of overfitting (train loss << val loss)
- Track wall-clock time per epoch for planning
- Save hyperparameters with each checkpoint
- Consider logging sample predictions during training

### Dependencies
- User Story 3.2 (Training Configuration)

---

## User Story 3.4: Validation During Training

**As a** ML engineer
**I want to** run validation evaluation during training
**So that** I can detect overfitting and select the best model checkpoint

### Acceptance Criteria
- [ ] Validation loop runs after each epoch
- [ ] Validation metrics computed:
  - Perplexity on validation set
  - Loss (cross-entropy)
  - Sample generation quality (qualitative spot checks)
- [ ] Validation results logged to experiment tracker
- [ ] Best checkpoint saved based on validation perplexity
- [ ] Validation time per epoch tracked
- [ ] Early stopping trigger configured (patience=2-3 epochs)
- [ ] Validation predictions saved for each checkpoint

### Technical Considerations
- Use `model.eval()` mode during validation
- Disable dropout during validation
- Generate 5-10 sample responses per validation run
- Compare validation metrics to baseline
- Monitor for divergence between train and validation metrics
- Consider computing ROUGE scores on validation set (optional, time permitting)

### Dependencies
- User Story 3.3 (Training Loop)
- Phase 1 validation data split

---

## User Story 3.5: Model Training Execution

**As a** ML engineer
**I want to** execute the full training run on Lightning AI or equivalent GPU
**So that** I produce a finetuned model ready for evaluation

### Acceptance Criteria
- [ ] Training environment set up on Lightning AI (T4 16GB) or equivalent
- [ ] All dependencies installed and verified
- [ ] Data uploaded or accessible from training environment
- [ ] Training script executed successfully
- [ ] Training completes without OOM errors
- [ ] Best checkpoint identified based on validation loss
- [ ] Training logs and checkpoints downloaded locally
- [ ] Total training time documented
- [ ] Final training curves plotted (loss, learning rate, etc.)
- [ ] Model adapter weights saved in Hugging Face format

### Technical Considerations
- Monitor GPU utilization throughout training
- Watch for memory leaks or gradual memory increase
- Have contingency plan for training interruptions (checkpointing)
- Consider running short test training run (1 epoch) first
- Document any training anomalies (loss spikes, NaN gradients)
- Save training script and full configuration with checkpoints

### Dependencies
- User Story 3.4 (Validation During Training)
- Lightning AI account setup

---

## User Story 3.6: Hyperparameter Experimentation (Optional)

**As a** ML engineer
**I want to** experiment with different hyperparameter configurations
**So that** I optimize model performance within time and resource constraints

### Acceptance Criteria
- [ ] At least 2-3 configurations tested with variations in:
  - Learning rate (1e-4, 2e-4, 3e-4)
  - LoRA rank (8, 16)
  - Batch size / gradient accumulation
  - Training epochs (3, 4, 5)
- [ ] Experiment tracking set up for all runs
- [ ] Comparison table created across experiments:
  - Final validation loss
  - Best validation perplexity
  - Training time
  - Memory usage
- [ ] Best configuration selected based on validation metrics
- [ ] Diminishing returns identified (when to stop experimenting)
- [ ] Experimentation findings documented

### Technical Considerations
- Use systematic approach (grid search or one-at-a-time variation)
- Keep most parameters constant while varying one
- Document rationale for each experiment
- Consider using smaller data subset for rapid iteration
- Don't over-optimize to validation set
- Balance performance improvement against training cost

### Dependencies
- User Story 3.5 (Initial Training Execution)

---

## User Story 3.7: Training Convergence Analysis

**As a** ML engineer
**I want to** analyze training convergence and learning dynamics
**So that** I verify the model learned effectively and identify potential issues

### Acceptance Criteria
- [ ] Training curves analyzed:
  - Loss convergence (smooth decline, plateau, or oscillation)
  - Validation vs training loss (overfitting check)
  - Learning rate schedule effectiveness
- [ ] Gradient statistics analyzed:
  - Gradient norms over time
  - Presence of vanishing/exploding gradients
  - Gradient flow to LoRA adapters
- [ ] Per-layer weight updates analyzed (if applicable)
- [ ] Convergence report created with:
  - Training curves plots
  - Identification of optimal epoch
  - Evidence of learning (improvement over baseline)
  - Recommendations for future training
- [ ] Failure modes documented if convergence issues found

### Technical Considerations
- Use logged metrics from Weights & Biases or TensorBoard
- Plot smoothed curves for better readability
- Compare final perplexity to baseline
- Check if more epochs would help or lead to overfitting
- Analyze whether model is underfitting (high train loss) or overfitting
- Consider visualizing attention patterns or embeddings (advanced)

### Dependencies
- User Story 3.5 (Training Execution)

---

## User Story 3.8: Model Adapter Export and Versioning

**As a** ML engineer
**I want to** export the trained LoRA adapters and version them properly
**So that** the finetuned model can be loaded for evaluation and future use

### Acceptance Criteria
- [ ] Best checkpoint identified from validation metrics
- [ ] LoRA adapter weights extracted and saved
- [ ] Model saved in Hugging Face format:
  - `adapter_model.bin` or `adapter_model.safetensors`
  - `adapter_config.json`
  - `tokenizer` files
  - `training_args.json`
- [ ] Model versioning implemented:
  - Version number assigned (e.g., v1.0)
  - Training date and metrics documented
  - Git commit hash recorded
- [ ] Model pushed to Hugging Face Hub (optional, for portfolio)
- [ ] Loading script created to merge adapters with base model
- [ ] README created for model card with:
  - Base model information
  - Training dataset
  - Hyperparameters
  - Performance metrics

### Technical Considerations
- Use `model.save_pretrained()` for Hugging Face format
- Test loading adapters back with `peft.PeftModel.from_pretrained()`
- Consider saving both merged and adapter-only versions
- Document inference requirements (memory, dependencies)
- Include reproducibility information (random seeds, versions)
- Consider model quantization for efficient deployment

### Dependencies
- User Story 3.5 (Training Execution)
- User Story 3.7 (Convergence Analysis)

---

## Phase 3 Definition of Done

All user stories (3.1-3.8) must be completed with:
- [ ] Unit tests implemented and passing
- [ ] Code formatted with `ruff format .`
- [ ] Type hints on all public functions
- [ ] Docstrings added with Args, Returns, Raises
- [ ] No regressions in existing tests
- [ ] README updated with training instructions and results
- [ ] Changes committed to git with clear commit messages

## Phase 3 Success Metrics

- Model trains successfully without OOM errors on T4 16GB
- Training converges with clear improvement over random initialization
- Validation loss decreases and plateaus appropriately
- Final validation perplexity is lower than baseline model
- Training completes in reasonable time (< 4 hours)
- LoRA adapters are properly saved and can be reloaded
- All training artifacts are versioned and documented
- Training process is reproducible from saved configurations
