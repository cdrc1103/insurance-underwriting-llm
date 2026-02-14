"""Training setup and execution for finetuning."""

import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from configs.model import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EVAL_STEPS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_SAVE_STEPS,
    DEFAULT_SCHEDULER,
    DEFAULT_SEED,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def create_training_args(
    output_dir: str | Path,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    per_device_train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs: int = DEFAULT_NUM_EPOCHS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    lr_scheduler_type: str = DEFAULT_SCHEDULER,
    logging_steps: int = DEFAULT_LOGGING_STEPS,
    save_steps: int = DEFAULT_SAVE_STEPS,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    seed: int = DEFAULT_SEED,
    fp16: bool = True,
    bf16: bool = False,
    logging_dir: str | Path | None = None,
) -> TrainingArguments:
    """
    Create TrainingArguments for the Trainer.

    Args:
        output_dir: Directory for checkpoints and outputs
        learning_rate: Peak learning rate
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_train_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for LR scheduler
        weight_decay: Weight decay for AdamW
        max_grad_norm: Maximum gradient norm for clipping
        lr_scheduler_type: Learning rate scheduler type
        logging_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Run evaluation every N steps
        seed: Random seed
        fp16: Use fp16 mixed precision
        bf16: Use bf16 mixed precision
        logging_dir: TensorBoard log directory

    Returns:
        Configured TrainingArguments

    Raises:
        ValueError: If output_dir is not provided
    """
    output_dir = Path(output_dir)
    if logging_dir is None:
        logging_dir = str(output_dir / "logs")

    args = TrainingArguments(
        output_dir=str(output_dir),
        # Training hyperparameters
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        optim="adamw_torch",
        # Mixed precision
        fp16=fp16,
        bf16=bf16,
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        # Reproducibility
        seed=seed,
        data_seed=seed,
        # Performance
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        # Gradient checkpointing (enabled via LoRA setup)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Disable find_unused_parameters for LoRA
        ddp_find_unused_parameters=False,
        # Remove unused columns (metadata)
        remove_unused_columns=True,
    )

    logger.info(f"Training args created: output_dir={output_dir}")
    logger.info(
        f"  lr={learning_rate}, batch_size={per_device_train_batch_size}, "
        f"grad_accum={gradient_accumulation_steps}, epochs={num_train_epochs}"
    )
    return args


def compute_metrics(eval_preds) -> dict[str, float]:
    """
    Compute perplexity from evaluation predictions.

    Args:
        eval_preds: EvalPrediction with predictions and label_ids

    Returns:
        Dictionary with perplexity metric
    """
    logits = eval_preds.predictions
    labels = eval_preds.label_ids

    # Shift logits and labels for causal LM
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Compute cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits_tensor = torch.tensor(shift_logits, dtype=torch.float32)
    shift_labels_tensor = torch.tensor(shift_labels, dtype=torch.long)

    # Flatten
    loss = loss_fct(
        shift_logits_tensor.view(-1, shift_logits_tensor.size(-1)),
        shift_labels_tensor.view(-1),
    )

    # Mask padding (-100 labels)
    mask = shift_labels_tensor.view(-1) != -100
    avg_loss = loss[mask].mean().item()
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {"perplexity": perplexity}


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
    callbacks: list | None = None,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
) -> Trainer:
    """
    Create a configured Trainer for finetuning.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Tokenized training dataset
        eval_dataset: Tokenized evaluation dataset
        training_args: Training configuration
        callbacks: Additional callbacks
        early_stopping_patience: Epochs to wait before early stopping

    Returns:
        Configured Trainer ready for training
    """
    all_callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    if callbacks:
        all_callbacks.extend(callbacks)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=all_callbacks,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Trainer created: {len(train_dataset)} train, {len(eval_dataset)} eval examples")
    return trainer


def run_training(trainer: Trainer, resume_from_checkpoint: str | Path | None = None):
    """
    Execute the training loop.

    Args:
        trainer: Configured Trainer
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        TrainOutput with training results

    Raises:
        RuntimeError: If training fails
    """
    logger.info("Starting training...")

    try:
        result = trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
        logger.info(
            f"Training complete: loss={result.training_loss:.4f}, steps={result.global_step}"
        )
        return result

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e
