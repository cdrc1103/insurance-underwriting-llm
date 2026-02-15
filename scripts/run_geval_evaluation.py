"""Run G-Eval evaluation on baseline model results.

Usage:
    # Default: Anthropic Claude with x samples
    python scripts/run_geval_evaluation.py

    # Custom model and samples
    python scripts/run_geval_evaluation.py --model gpt-4o --num-samples 10

    # Custom input/output paths
    python scripts/run_geval_evaluation.py \
        --input results/baseline_evaluation.json \
        --output results/geval_results.json

    # Resume from previous run
    python scripts/run_geval_evaluation.py --resume
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.criterion import CRITERIA
from src.evaluation.evaluator import compute_aggregate_metrics
from src.evaluation.judge_model_provider import create_provider
from src.evaluation.models import GEvalConfig
from src.evaluation.utils import (
    load_existing_results,
    load_input_data,
    log_summary,
    prepare_evaluation_batch,
    run_evaluation_loop,
)
from src.logging import setup_logging

# Configure logging to both console and file
log_file = setup_logging(
    log_dir=Path("logs"),
    log_prefix="geval_evaluation",
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Run G-Eval evaluation on baseline model results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/baseline_evaluation.json"),
        help="Path to baseline evaluation results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/geval_baseline_evaluation.json"),
        help="Path to save G-Eval results JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model identifier (e.g., claude-x, gpt-x)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per criterion for probability estimation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Number of output tokens allowed the generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for G-Eval",
    )
    parser.add_argument(
        "--sample-delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file if it exists",
    )
    return parser.parse_args()


def main() -> None:
    """Run the G-Eval evaluation pipeline with incremental saving and resume support."""
    load_dotenv()
    args = parse_args()

    # Load input data
    examples = load_input_data(args.input)

    # Determine output path with date
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = args.output.parent / f"{args.output.stem}_{date_str}{args.output.suffix}"

    # Load existing results if resuming
    existing_results, evaluated_ids = (
        load_existing_results(output_path) if args.resume else ([], set())
    )

    # Prepare batch to evaluate
    examples_to_evaluate = prepare_evaluation_batch(examples, evaluated_ids, args.max_examples)

    if not examples_to_evaluate:
        logger.info("No examples to evaluate. All work is complete!")
        return

    # Create provider and config
    provider = create_provider(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    config = GEvalConfig(
        num_samples=args.num_samples,
        temperature=args.temperature,
        sample_delay=args.sample_delay,
        max_tokens=args.max_tokens,
    )

    # Log configuration
    logger.info("Model: %s", args.model)
    logger.info("Samples per criterion: %d", config.num_samples)
    logger.info("Temperature: %.1f", config.temperature)
    logger.info("Max Tokens: %d", config.max_tokens)

    # Estimate cost
    estimated_calls = len(examples_to_evaluate) * len(CRITERIA) * config.num_samples
    logger.info("Estimated API calls for remaining examples: %d", estimated_calls)

    # Run evaluation with incremental saving
    all_results = run_evaluation_loop(
        examples_to_evaluate, provider, config, output_path, args.model, existing_results, CRITERIA
    )

    logger.info("Results saved to %s", output_path)

    # Log summary
    metrics = compute_aggregate_metrics(all_results)
    log_summary(metrics, args.model)


if __name__ == "__main__":
    main()
