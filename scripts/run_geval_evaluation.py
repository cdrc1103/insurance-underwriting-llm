"""Run G-Eval evaluation on baseline model results.

Usage:
    # Default: Anthropic Claude with 20 samples
    python scripts/run_geval_evaluation.py

    # Custom model and samples
    python scripts/run_geval_evaluation.py --model gpt-4o --num-samples 10

    # Custom input/output paths
    python scripts/run_geval_evaluation.py \
        --input results/baseline_evaluation.json \
        --output results/geval_results.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from configs.model import DEFAULT_MAX_NEW_TOKENS
from src.evaluation.judge_evaluator import (
    GEvalConfig,
    batch_evaluate,
    results_to_dict,
)
from src.evaluation.judge_model_provider import create_provider
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
    return parser.parse_args()


def main() -> None:
    """Run the G-Eval evaluation pipeline."""
    load_dotenv()
    args = parse_args()

    # Load input data
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    results = data["results"] if "results" in data else data
    logger.info("Loaded %d examples from %s", len(results), args.input)

    # Limit examples if requested
    if args.max_examples is not None:
        results = results[: args.max_examples]
        logger.info("Limited to %d examples", len(results))

    # Create provider and config
    provider = create_provider(
        model=args.model,
        temperature=args.temperature,
        max_tokens=DEFAULT_MAX_NEW_TOKENS,
    )

    config = GEvalConfig(
        num_samples=args.num_samples,
        temperature=args.temperature,
        sample_delay=args.sample_delay,
        max_tokens=args.max_tokens,
    )

    logger.info("Model: %s", args.model)
    logger.info("Samples per criterion: %d", config.num_samples)
    logger.info("Temperature: %.1f", config.temperature)
    logger.info("Max Tokens, %d", config.max_tokens)

    # Estimate cost
    estimated_calls = len(results) * 6 * config.num_samples
    logger.info("Estimated API calls: %d", estimated_calls)

    # Run evaluation
    geval_results = batch_evaluate(results, provider, config)

    # Serialize and save
    output_dict = results_to_dict(geval_results, args.model, config)

    # Add date identifier to output filename
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = args.output.parent / f"{args.output.stem}_{date_str}{args.output.suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Log summary
    metrics = output_dict["aggregate_metrics"]

    summary_lines = [
        "=== G-Eval Results Summary ===",
        f"Model: {args.model}",
        f"Examples: {metrics['total_examples']} ({metrics['valid_examples']} valid)",
        f"Overall Mean Score: {metrics['overall_mean']:.2f} / 5.0",
        f"Total API Calls: {metrics['total_api_calls']}",
        f"Total Cost: ${metrics['total_cost_usd']:.4f}",
        "\nBy Task Type:",
    ]
    for task, task_data in metrics.get("by_task", {}).items():
        summary_lines.append(f"  {task}: {task_data['mean']:.2f} (n={task_data['count']})")

    summary_lines.append("\nBy Criterion:")
    for criterion, crit_data in metrics.get("by_criterion", {}).items():
        summary_lines.append(f"  {criterion}: {crit_data['mean']:.2f} (n={crit_data['count']})")

    logger.info("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
