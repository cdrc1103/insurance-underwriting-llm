"""Utility functions for G-Eval evaluation."""

import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.models import GEvalConfig, GEvalResult

logger = logging.getLogger(__name__)


def parse_company_profile(system_message: str) -> dict[str, str]:
    """Extract company profile from the system message.

    Args:
        system_message: System message content containing company profile section

    Returns:
        Dictionary mapping profile field names to values

    Raises:
        ValueError: If no company profile section is found
    """
    match = re.search(r"## Company Profile\s*\n(.*?)(?:\n##|\Z)", system_message, re.DOTALL)
    if not match:
        raise ValueError("No '## Company Profile' section found in system message")

    profile: dict[str, str] = {}
    for line in match.group(1).strip().split("\n"):
        line = line.strip()
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            profile[key.strip()] = value.strip()

    return profile


def parse_score_from_response(response_text: str) -> int:
    """Extract score from a G-Eval judge response.

    Looks for "SCORE: X" pattern at the end of the response.

    Args:
        response_text: The judge model's response text

    Returns:
        Integer score (0-5)

    Raises:
        ValueError: If no valid score is found
    """
    match = re.search(r"SCORE:\s*(\d)", response_text)
    if not match:
        # Log full response for debugging when score pattern not found
        logger.debug("Full response text (length=%d): %s", len(response_text), response_text)
        raise ValueError(
            f"No 'SCORE: X' pattern found in response (length={len(response_text)}): "
            f"{response_text[:200] if response_text else '[EMPTY]'}"
        )

    score = int(match.group(1))
    if score < 0 or score > 5:
        raise ValueError(f"Score {score} out of valid range 0-5")

    return score


def compute_weighted_score(score_probabilities: dict[int, float]) -> float:
    """Compute G-Eval probability-weighted score.

    Calculates: Σ p(score_i) × score_i

    Args:
        score_probabilities: Mapping of score values to their probabilities

    Returns:
        Weighted score as a float

    Raises:
        ValueError: If score_probabilities is empty
    """
    if not score_probabilities:
        raise ValueError("score_probabilities must not be empty")

    return sum(score * prob for score, prob in score_probabilities.items())


def compute_score_probabilities(raw_scores: list[int]) -> dict[int, float]:
    """Compute probability distribution from raw sample scores.

    Args:
        raw_scores: List of integer scores from multiple samples

    Returns:
        Dictionary mapping each score to its probability

    Raises:
        ValueError: If raw_scores is empty
    """
    if not raw_scores:
        raise ValueError("raw_scores must not be empty")

    counts = Counter(raw_scores)
    total = len(raw_scores)
    return {score: count / total for score, count in sorted(counts.items())}


def load_existing_results(output_path: Path) -> tuple[list[GEvalResult], set[int]]:
    """Load existing evaluation results from output file.

    Args:
        output_path: Path to the results JSON file

    Returns:
        Tuple of (list of GEvalResult objects, set of evaluated example IDs)
    """
    if not output_path.exists():
        return [], set()

    try:
        with output_path.open() as f:
            data = json.load(f)

        results = [GEvalResult.from_dict(r) for r in data.get("results", [])]
        evaluated_ids = {r.example_id for r in results}

        logger.info("Loaded %d existing results from %s", len(results), output_path)
        return results, evaluated_ids

    except Exception as e:
        logger.warning(
            "Failed to load existing results from %s: %s. Starting fresh.",
            output_path,
            e,
        )
        return [], set()


def build_output_dict(
    results: list[GEvalResult],
    model: str,
    config: GEvalConfig,
) -> dict[str, Any]:
    """Build output dictionary for JSON serialization.

    Args:
        results: List of GEvalResult objects
        model: Model identifier used
        config: G-Eval configuration used

    Returns:
        Dictionary ready for JSON serialization
    """
    from src.evaluation.evaluator import compute_aggregate_metrics

    return {
        "evaluation_metadata": {
            "model": model,
            "num_samples": config.num_samples,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "total_examples": len(results),
            "total_api_calls": sum(r.total_api_calls for r in results),
            "total_cost_usd": sum(r.total_cost_usd for r in results),
            "last_updated": datetime.now().isoformat(),
        },
        "results": [r.to_dict() for r in results],
        "aggregate_metrics": compute_aggregate_metrics(results),
    }


def save_results_incrementally(
    output_path: Path,
    all_results: list[GEvalResult],
    model: str,
    config: GEvalConfig,
) -> None:
    """Save all results to JSON file with metadata and aggregate metrics.

    Uses atomic write (temp file + rename) to prevent corruption.

    Args:
        output_path: Path to save results
        all_results: List of all GEvalResult objects
        model: Model identifier used
        config: G-Eval configuration used
    """
    output_dict = build_output_dict(all_results, model, config)

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically using temp file
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w") as f:
        json.dump(output_dict, f, indent=2)

    # Replace original file
    temp_path.replace(output_path)


def load_input_data(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate input examples.

    Args:
        input_path: Path to input JSON file

    Returns:
        List of example dictionaries

    Raises:
        SystemExit: If file not found
    """
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    with input_path.open() as f:
        data = json.load(f)

    examples: list[dict[str, Any]] = data["results"] if "results" in data else data
    logger.info("Loaded %d examples from %s", len(examples), input_path)
    return examples


def prepare_evaluation_batch(
    examples: list[dict[str, Any]],
    evaluated_ids: set[int],
    max_examples: int | None,
) -> list[dict[str, Any]]:
    """Filter examples to evaluate based on resume state.

    Args:
        examples: All available examples
        evaluated_ids: Set of already-evaluated example IDs
        max_examples: Maximum number of examples to process (None for all)

    Returns:
        Filtered list of examples to evaluate
    """
    # Filter out already-evaluated examples
    examples_to_evaluate = [
        ex for ex in examples if ex.get("original_index", -1) not in evaluated_ids
    ]

    if evaluated_ids:
        logger.info(
            "Skipping %d already-evaluated examples, %d remaining",
            len(evaluated_ids),
            len(examples_to_evaluate),
        )

    # Limit if requested
    if max_examples is not None:
        examples_to_evaluate = examples_to_evaluate[:max_examples]
        logger.info("Limited to %d examples for this run", len(examples_to_evaluate))

    return examples_to_evaluate


def run_evaluation_loop(
    examples: list[dict[str, Any]],
    provider: Any,
    config: GEvalConfig,
    output_path: Path,
    model: str,
    existing_results: list[GEvalResult],
    criteria: list[Any],
) -> list[GEvalResult]:
    """Run evaluation loop with incremental saving.

    Args:
        examples: Examples to evaluate
        provider: Model provider for judge calls
        config: G-Eval configuration
        output_path: Path to save results
        model: Model identifier
        existing_results: Previously evaluated results
        criteria: Evaluation criteria to use

    Returns:
        Complete list of all results (existing + new)
    """
    from src.evaluation.evaluator import evaluate_example

    all_results = existing_results.copy()

    for example in tqdm(examples, desc="Evaluating examples"):
        example_id = example.get("original_index", "?")
        logger.info("Evaluating example %s (task: %s)", example_id, example.get("task"))

        try:
            result = evaluate_example(provider, example, criteria, config)
            all_results.append(result)

            logger.info(
                "  Example %s: overall=%.2f, cost=$%.4f",
                example_id,
                result.overall_score,
                result.total_cost_usd,
            )

        except Exception as e:
            logger.error("Failed to evaluate example %s: %s", example_id, e)

            # Save failed result to avoid re-processing
            failed_result = GEvalResult(
                example_id=example.get("original_index", 0),
                task_type=example.get("task", "unknown"),
                criterion_scores=[],
                overall_score=0.0,
                total_cost_usd=0.0,
                total_api_calls=0,
                evaluation_time_ms=0.0,
            )
            all_results.append(failed_result)

        # Save incrementally after each example
        save_results_incrementally(output_path, all_results, model, config)

    return all_results


def log_summary(metrics: dict[str, Any], model: str) -> None:
    """Log evaluation summary statistics.

    Args:
        metrics: Aggregate metrics dictionary
        model: Model identifier
    """
    summary_lines = [
        "=== G-Eval Results Summary ===",
        f"Model: {model}",
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
