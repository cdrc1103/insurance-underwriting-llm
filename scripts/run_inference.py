#!/usr/bin/env python3
"""Run inference on a dataset using a trained model.

This script loads a model and runs inference on a dataset of insurance
underwriting conversations, generating responses and saving results.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent if __name__ == "__main__" else Path.cwd()
sys.path.insert(0, str(project_root))

from src.evaluation.inference import (
    GenerationConfig,
    evaluate_dataset,
    evaluate_dataset_batched,
    save_evaluation_results,
)
from src.models.model_loader import load_base_model


def load_dataset(dataset_path: Path) -> list[dict]:
    """
    Load dataset from JSON file.

    Args:
        dataset_path: Path to dataset JSON file

    Returns:
        List of dataset examples

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset format is invalid
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of examples")

    return dataset


def run_inference(
    model_path: str,
    dataset_path: Path,
    output_path: Path,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int | None = None,
    device: str = "auto",
) -> None:
    """
    Run inference on dataset and save results.

    Args:
        model_path: Path or name of model to load
        dataset_path: Path to dataset JSON file
        output_path: Path to save results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for batched inference (None for sequential)
        device: Device to run on ('auto', 'cuda', 'cpu')
    """
    print("=" * 80)
    print("Running Inference")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"  Loaded {len(dataset)} examples")
    print()

    # Load model
    print("Loading model and tokenizer...")
    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    model, tokenizer = load_base_model(
        model_name=model_path,
        device_map=device,
        torch_dtype=torch_dtype,
    )
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")
    print(f"  Device: {next(model.parameters()).device}")
    print()

    # Configure generation
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("Generation configuration:")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Top-p: {config.top_p}")
    print(f"  Top-k: {config.top_k}")
    print(f"  Repetition penalty: {config.repetition_penalty}")
    print()

    # Run evaluation
    if batch_size is not None:
        print(f"Running batched inference (batch_size={batch_size})...")
        print()
        result = evaluate_dataset_batched(
            model, tokenizer, dataset, config=config, batch_size=batch_size
        )
    else:
        print("Running sequential inference...")
        print()
        result = evaluate_dataset(model, tokenizer, dataset, config=config)

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Successful: {result.successful_count}")
    print(f"Failed: {result.failed_count}")
    print(f"Total time: {result.total_time_ms:.2f}ms")
    print(f"Average time per example: {result.total_time_ms / len(dataset):.2f}ms")
    print()

    # Calculate token statistics
    total_input_tokens = sum(
        r["input_tokens"] for r in result.results if r.get("input_tokens") is not None
    )
    total_output_tokens = sum(
        r["output_tokens"] for r in result.results if r.get("output_tokens") is not None
    )
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Average input tokens: {total_input_tokens / len(dataset):.1f}")
    print(f"Average output tokens: {total_output_tokens / len(dataset):.1f}")
    print()

    # Save results
    print(f"Saving results to {output_path}...")
    save_evaluation_results(result, output_path)
    print("Done!")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run inference on insurance underwriting dataset")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen3-0.6B' or path to fine-tuned model)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, use 0.0 for greedy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for batched inference (default: None for sequential)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on (default: auto)",
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
