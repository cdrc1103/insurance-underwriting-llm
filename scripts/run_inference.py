#!/usr/bin/env python3
"""Run inference on a dataset using a trained model.

This script loads a model and runs inference on a dataset of insurance
underwriting conversations, generating responses and saving results.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

from configs.model import DEFAULT_MODEL_NAME
from src.data.dataset_io import load_dataset_split
from src.evaluation.inference import (
    GenerationConfig,
    evaluate_dataset,
    evaluate_dataset_batched,
    save_evaluation_results,
)
from src.models.model_loader import load_base_model

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

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"inference_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_file


def run_inference(
    model_path: str,
    dataset_path: Path,
    output_path: Path,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int | None = None,
    device: str = "auto",
    log_dir: Path = Path("logs"),
) -> None:
    """
    Run inference on dataset and save results.

    Args:
        model_path: Path or name of model to load
        dataset_path: Path to dataset directory (Arrow format)
        output_path: Path to save results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for batched inference (None for sequential)
        device: Device to run on ('auto', 'cuda', 'cpu')
        log_dir: Directory to save log files (default: logs/)
    """
    # Set up logging
    log_file = setup_logging(log_dir)
    logger.info("=" * 80)
    logger.info("Running Inference")
    logger.info("=" * 80)
    logger.info(f"\nModel: {model_path}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {device}")
    logger.info("")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = list(load_dataset_split(dataset_path.parent, dataset_path.name))
    logger.info(f"  Loaded {len(dataset)} examples")
    logger.info("")

    # Load model
    logger.info("Loading model and tokenizer...")
    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    model, tokenizer = load_base_model(
        model_name=model_path,
        device_map=device,
        torch_dtype=torch_dtype,
    )
    logger.info(f"  Model: {model.__class__.__name__}")
    logger.info(f"  Tokenizer: {tokenizer.__class__.__name__}")
    logger.info(f"  Device: {next(model.parameters()).device}")
    logger.info("")

    # Configure generation
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    logger.info("Generation configuration:")
    logger.info(f"  Max new tokens: {config.max_new_tokens}")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Top-p: {config.top_p}")
    logger.info(f"  Top-k: {config.top_k}")
    logger.info(f"  Repetition penalty: {config.repetition_penalty}")
    logger.info("")

    # Run evaluation
    if batch_size is not None:
        logger.info(f"Running batched inference (batch_size={batch_size})...")
        logger.info("")
        result = evaluate_dataset_batched(
            model, tokenizer, dataset, config=config, batch_size=batch_size
        )
    else:
        logger.info("Running sequential inference...")
        logger.info("")
        result = evaluate_dataset(model, tokenizer, dataset, config=config)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Results")
    logger.info("=" * 80)
    logger.info(f"Successful: {result.successful_count}")
    logger.info(f"Failed: {result.failed_count}")
    logger.info(f"Total time: {result.total_time_ms:.2f}ms")
    logger.info(f"Average time per example: {result.total_time_ms / len(dataset):.2f}ms")
    logger.info("")

    # Calculate token statistics
    total_input_tokens = sum(
        r["input_tokens"] for r in result.results if r.get("input_tokens") is not None
    )
    total_output_tokens = sum(
        r["output_tokens"] for r in result.results if r.get("output_tokens") is not None
    )
    logger.info(f"Total input tokens: {total_input_tokens:,}")
    logger.info(f"Total output tokens: {total_output_tokens:,}")
    logger.info(f"Average input tokens: {total_input_tokens / len(dataset):.1f}")
    logger.info(f"Average output tokens: {total_output_tokens / len(dataset):.1f}")
    logger.info("")

    # Save results
    logger.info(f"Saving results to {output_path}...")
    save_evaluation_results(result, output_path)
    logger.info("Done!")
    logger.info("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run inference on insurance underwriting dataset")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen3-0.6B' or path to fine-tuned model)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/splits/test",
        required=True,
        help="Path to dataset directory (Hugging Face Arrow format)",
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
    parser.add_argument(
        "--log-dir",
        type=Path,
        default="logs",
        help="Directory to save log files (default: logs/)",
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
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
