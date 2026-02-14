#!/usr/bin/env python3
"""Run LoRA finetuning on the insurance underwriting dataset.

This script loads the base model, applies LoRA adapters, and trains
on the preprocessed dataset with label masking and validation.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Set PyTorch memory allocator configuration BEFORE importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from configs.model import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EVAL_STEPS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_SAVE_STEPS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
    MAX_TOKEN_LENGTH,
)
from src.data.dataset_io import load_dataset_split
from src.models.model_loader import load_base_model, profile_model_memory
from src.training.callbacks import MemoryLoggingCallback, SampleGenerationCallback
from src.training.data import prepare_sft_dataset
from src.training.export import export_adapter
from src.training.lora_config import (
    apply_lora_adapters,
    create_lora_config,
    create_quantization_config,
    get_trainable_parameters,
)
from src.training.trainer import (
    create_trainer,
    create_training_args,
    run_training,
    set_seed,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> Path:
    """
    Set up logging to both console and file.

    Args:
        log_dir: Directory to save log files

    Returns:
        Path to the created log file
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file


def run_finetuning(
    model_name: str,
    train_data_path: Path,
    val_data_path: Path,
    output_dir: Path,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    max_seq_length: int,
    use_qlora: bool,
    seed: int,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    early_stopping_patience: int,
    log_dir: Path,
    resume_from: str | None,
) -> None:
    """
    Execute the full finetuning pipeline.

    Args:
        model_name: HuggingFace model identifier
        train_data_path: Path to training data split
        val_data_path: Path to validation data split
        output_dir: Directory for checkpoints and final model
        learning_rate: Peak learning rate
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        warmup_steps: LR warmup steps
        weight_decay: AdamW weight decay
        max_grad_norm: Gradient clipping norm
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        max_seq_length: Maximum sequence length for tokenization
        use_qlora: Whether to use QLoRA 4-bit quantization
        seed: Random seed
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        early_stopping_patience: Early stopping patience
        log_dir: Directory for log files
        resume_from: Path to checkpoint to resume from
    """
    log_file = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("LoRA Finetuning - Insurance Underwriting")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Train data: {train_data_path}")
    logger.info(f"Val data: {val_data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"QLoRA: {use_qlora}")
    logger.info("")

    # Set seed
    set_seed(seed)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset_split(train_data_path.parent, train_data_path.name)
    val_dataset = load_dataset_split(val_data_path.parent, val_data_path.name)
    logger.info(f"  Train: {len(train_dataset)} examples")
    logger.info(f"  Val: {len(val_dataset)} examples")
    logger.info("")

    # Load model
    logger.info("Loading base model...")
    model_kwargs: dict = {}
    if use_qlora:
        model_kwargs["quantization_config"] = create_quantization_config()
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    model, tokenizer = load_base_model(
        model_name=model_name,
        device_map="auto",
        dtype=dtype,
        **model_kwargs,
    )

    memory_info = profile_model_memory(model)
    logger.info(f"  Model size: {memory_info['model_size_gb']:.3f} GB")
    logger.info(f"  Device: {memory_info['device']}")
    logger.info("")

    # Apply LoRA adapters
    logger.info("Applying LoRA adapters...")
    lora_config = create_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = apply_lora_adapters(model, lora_config)

    param_info = get_trainable_parameters(model)
    logger.info(
        f"  Trainable: {param_info['trainable_params']:,} ({param_info['trainable_percent']:.2f}%)"
    )
    logger.info(f"  Total: {param_info['total_params']:,}")
    logger.info("")

    # Prepare SFT datasets
    logger.info("Preparing SFT datasets with label masking...")
    train_tokenized = prepare_sft_dataset(train_dataset, tokenizer, max_length=max_seq_length)
    val_tokenized = prepare_sft_dataset(val_dataset, tokenizer, max_length=max_seq_length)
    logger.info(f"  Train tokenized: {len(train_tokenized)} examples")
    logger.info(f"  Val tokenized: {len(val_tokenized)} examples")
    logger.info("")

    # Create training args
    training_args = create_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        seed=seed,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )

    # Create callbacks
    # Get a few sample prompts from validation for qualitative monitoring
    sample_prompts = [
        example["messages"] for example in val_dataset.select(range(min(3, len(val_dataset))))
    ]
    callbacks = [
        MemoryLoggingCallback(),
        SampleGenerationCallback(
            tokenizer=tokenizer,
            sample_prompts=sample_prompts,
        ),
    ]

    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        training_args=training_args,
        callbacks=callbacks,
        early_stopping_patience=early_stopping_patience,
    )

    # Run training
    logger.info("Starting training...")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(
        f"  Total steps: ~{len(train_tokenized) * num_epochs // (batch_size * gradient_accumulation_steps)}"
    )
    logger.info("")

    result = run_training(trainer, resume_from_checkpoint=resume_from)

    # Log final results
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Results")
    logger.info("=" * 80)
    logger.info(f"  Final loss: {result.training_loss:.4f}")
    logger.info(f"  Total steps: {result.global_step}")
    logger.info("")

    # Export best model
    final_model_dir = output_dir / "final_adapter"
    logger.info(f"Exporting best model to {final_model_dir}...")

    training_config = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_seq_length": max_seq_length,
        "use_qlora": use_qlora,
        "seed": seed,
    }

    metrics = {
        "training_loss": result.training_loss,
        "global_step": result.global_step,
    }

    export_adapter(
        model=trainer.model,
        tokenizer=tokenizer,
        output_dir=final_model_dir,
        training_config=training_config,
        metrics=metrics,
    )

    # Save full training config for reproducibility
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    logger.info(f"Training config saved to {config_path}")

    logger.info("")
    logger.info("Done!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Finetune Qwen3-0.6B with LoRA for insurance underwriting"
    )

    # Data arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Base model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/splits/train"),
        help="Path to training data split",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/splits/validation"),
        help="Path to validation data split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/finetuned"),
        help="Output directory for checkpoints",
    )

    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--max-seq-length", type=int, default=MAX_TOKEN_LENGTH)

    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--use-qlora", action="store_true", help="Enable QLoRA 4-bit quantization")

    # Logging and checkpointing
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )

    # Other
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    run_finetuning(
        model_name=args.model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        use_qlora=args.use_qlora,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
