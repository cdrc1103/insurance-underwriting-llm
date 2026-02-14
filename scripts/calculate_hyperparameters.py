#!/usr/bin/env python3
"""PyTorch hyperparameter calculator.

This script calculates optimal hyperparameters
for custom PyTorch training based on GPU memory constraints.
"""

import argparse
import json
from pathlib import Path

from configs.model import MAX_TOKEN_LENGTH
from src.data.splitting import load_splits
from src.data.tokenization import load_tokenizer


def calculate_memory_per_example(
    seq_length: int,
    num_layers: int = 24,
) -> float:
    """
    Calculate memory (GB) required per example for activations.

    Args:
        seq_length: Sequence length in tokens
        num_layers: Number of transformer layers (default: 24 for Qwen3-0.6B)

    Returns:
        Memory in GB
    """
    # Activations: batch_size=1, seq_length, hidden_dim, num_layers
    # Approximate: 12 bytes per token per layer (includes all intermediate activations)
    activation_memory_bytes = seq_length * num_layers * 12
    return activation_memory_bytes / (1024**3)


def calculate_optimal_batch_size(
    gpu_memory_gb: float,
    model_memory_gb: float,
    seq_length: int,
    safety_margin: float = 0.85,
) -> int:
    """Calculate optimal batch size that fits in GPU memory."""
    available_memory = (gpu_memory_gb - model_memory_gb) * safety_margin
    memory_per_example = calculate_memory_per_example(seq_length)

    batch_size = max(1, int(available_memory / memory_per_example))
    return min(batch_size, 4)  # Cap at 4 for stability


def calculate_pytorch_hyperparameters(
    max_seq_length: int,
    num_train_examples: int,
    gpu_memory_gb: float = 16.0,
    model_params_millions: float = 619.0,  # Qwen3-0.6B
) -> dict:
    """Calculate optimal PyTorch training hyperparameters."""
    print("\n" + "=" * 80)
    print(f"PyTorch Training Hyperparameters (GPU: {gpu_memory_gb}GB)")
    print("=" * 80)

    # Model memory estimates
    use_lora = True
    # LoRA rank: dimensionality of low-rank adaptation matrices (higher = more parameters)
    lora_r = 16

    # Base model in FP16
    model_memory_gb = (model_params_millions * 2) / 1024

    # LoRA params (much smaller than full model)
    # Qwen3-0.6B: ~24 layers, d_model=896, 7 target modules
    lora_params_millions = (2 * lora_r * 896 * 24 * 7) / 1e6

    # Optimizer states (AdamW: 8 bytes per trainable param)
    optimizer_memory_gb = (lora_params_millions * 8) / 1024

    # Calculate optimal batch size
    batch_size = calculate_optimal_batch_size(
        gpu_memory_gb, model_memory_gb + optimizer_memory_gb, max_seq_length
    )

    # Gradient accumulation for effective batch size
    target_effective_batch = 8
    grad_accum_steps = max(1, target_effective_batch // batch_size)
    effective_batch_size = batch_size * grad_accum_steps

    # Training schedule
    num_epochs = 3
    steps_per_epoch = num_train_examples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.1)

    # Learning rate (scaled with effective batch size)
    base_lr = 2e-4
    learning_rate = base_lr * (effective_batch_size / 8) ** 0.5

    # Memory estimate
    activation_memory_per_batch = calculate_memory_per_example(max_seq_length) * batch_size
    total_memory_gb = model_memory_gb + optimizer_memory_gb + activation_memory_per_batch

    config = {
        # Model configuration
        "model_name": "Qwen/Qwen3-0.6B",
        "model_params_millions": model_params_millions,
        "max_seq_length": max_seq_length,
        # Batch configuration
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        # Training schedule
        "num_train_epochs": num_epochs,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "warmup_ratio": 0.1,
        # Optimizer (PyTorch AdamW)
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        # Scheduler (Cosine with warmup)
        "scheduler": "cosine",
        "min_lr_ratio": 0.1,
        # Mixed precision
        "use_fp16": True,
        "use_bf16": False,  # T4 doesn't support bf16
        # LoRA configuration
        "use_lora": use_lora,
        "lora_r": lora_r,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_trainable_params_millions": lora_params_millions,
        # Memory estimates
        "memory_breakdown_gb": {
            "model": model_memory_gb,
            "optimizer": optimizer_memory_gb,
            "activations_per_batch": activation_memory_per_batch,
            "total_estimated": total_memory_gb,
        },
        "gpu_memory_gb": gpu_memory_gb,
        "memory_utilization_percent": (total_memory_gb / gpu_memory_gb) * 100,
    }

    # Print configuration
    print("\nModel Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Parameters: {config['model_params_millions']:.1f}M")
    print(f"  Max sequence length: {config['max_seq_length']:,} tokens")

    print("\nLoRA Configuration:")
    print(f"  Enabled: {config['use_lora']}")
    print(f"  Rank (r): {config['lora_r']}")
    print(f"  Alpha: {config['lora_alpha']}")
    print(f"  Dropout: {config['lora_dropout']}")
    print(f"  Target modules: {len(config['lora_target_modules'])} modules")
    print(f"  Trainable params: {config['lora_trainable_params_millions']:.2f}M")

    print("\nBatch Configuration:")
    print(f"  Per-device batch size: {config['per_device_train_batch_size']}")
    print(f"  Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {config['effective_batch_size']}")

    print("\nTraining Schedule:")
    print(f"  Number of epochs: {config['num_train_epochs']}")
    print(f"  Steps per epoch: {config['steps_per_epoch']}")
    print(f"  Total training steps: {config['total_steps']}")
    print(f"  Warmup steps: {config['warmup_steps']} ({config['warmup_ratio'] * 100:.0f}%)")

    print("\nPyTorch Optimizer (AdamW):")
    print(f"  Learning rate: {config['learning_rate']:.2e}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Betas: ({config['adam_beta1']}, {config['adam_beta2']})")
    print(f"  Epsilon: {config['adam_epsilon']}")
    print(f"  Max grad norm: {config['max_grad_norm']}")

    print("\nPyTorch Scheduler:")
    print(f"  Type: {config['scheduler']}")
    print(f"  Min LR ratio: {config['min_lr_ratio']}")

    print("\nMixed Precision:")
    print(f"  FP16: {config['use_fp16']}")
    print(f"  BF16: {config['use_bf16']}")

    mem = config["memory_breakdown_gb"]
    print("\nMemory Estimates:")
    print(f"  Model (FP16): {mem['model']:.2f} GB")
    print(f"  Optimizer states: {mem['optimizer']:.2f} GB")
    print(f"  Activations (per batch): {mem['activations_per_batch']:.2f} GB")
    print(f"  Total estimated: {mem['total_estimated']:.2f} GB")
    print(f"  GPU memory: {config['gpu_memory_gb']:.1f} GB")
    print(f"  Utilization: {config['memory_utilization_percent']:.1f}%")
    print()

    return config


def calculate_hyperparameters(
    splits_dir: Path,
    model_name: str = "Qwen/Qwen3-0.6B",
    max_seq_length: int = MAX_TOKEN_LENGTH,
    output_file: Path | None = None,
) -> dict:
    """Hyperparameter calculation for PyTorch training."""
    print("=" * 80)
    print("Hyperparameter calculation for PyTorch Training")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Splits directory: {splits_dir}")
    print(f"Max sequence length: {max_seq_length:,}")
    print()

    # Load tokenizer and splits
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")
    print()

    print("Loading dataset splits...")
    splits = load_splits(splits_dir)
    print()

    # Verify data loaded successfully
    print("Verifying preprocessed data...")
    print("-" * 80)
    for split_name in ["train", "validation", "test"]:
        if split_name in splits:
            print(f"  ✓ {split_name.capitalize()} split: {len(splits[split_name])} examples")
    print()

    # Calculate PyTorch hyperparameters
    pytorch_config = calculate_pytorch_hyperparameters(
        max_seq_length=max_seq_length,
        num_train_examples=len(splits["train"]),
    )

    # Compile results
    results = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "pytorch_training_config": pytorch_config,
    }
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

    print()
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Calculate PyTorch hyperparameters")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name for tokenizer",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_TOKEN_LENGTH,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hyperparameters.json"),
        help="Output file for hyperparameter results",
    )

    args = parser.parse_args()

    calculate_hyperparameters(
        splits_dir=args.splits_dir,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
