"""High-level evaluation functions for G-Eval."""

import asyncio
import logging
import time
from typing import Any

from tqdm import tqdm

from src.evaluation.criterion import CRITERIA, EvaluationCriterion
from src.evaluation.geval_scorer import score_criterion
from src.evaluation.judge_model_provider import LiteLLMProvider
from src.evaluation.models import CriterionScore, GEvalConfig, GEvalResult
from src.evaluation.utils import parse_company_profile

logger = logging.getLogger(__name__)


def evaluate_example(
    provider: LiteLLMProvider,
    example: dict[str, Any],
    criteria: list[EvaluationCriterion],
    config: GEvalConfig,
) -> GEvalResult:
    """Evaluate a single example across all criteria.

    Args:
        provider: Model provider for judge calls
        example: Example dict from baseline_evaluation.json
        criteria: List of evaluation criteria
        config: G-Eval configuration

    Returns:
        GEvalResult with scores for all criteria

    Raises:
        ValueError: If example is missing required fields
    """
    start_time = time.time()

    # Validate required fields
    required_fields = ["messages", "generated_response", "target_response"]
    missing = [f for f in required_fields if f not in example]
    if missing:
        raise ValueError(f"Example missing required fields: {missing}")
    if not example["messages"] or not isinstance(example["messages"], list):
        raise ValueError("messages must be a non-empty list")

    # Parse inputs
    messages = example["messages"]
    system_message = messages[0]["content"]
    company_profile = parse_company_profile(system_message)
    conversation = messages[1:]
    generated_response = example["generated_response"]
    reference_answer = example["target_response"]

    # Score all criteria in parallel
    logger.info("  Scoring %d criteria in parallel", len(criteria))

    async def score_all_criteria() -> list[CriterionScore | BaseException]:
        """Score all criteria concurrently with exception handling."""
        tasks = [
            score_criterion(
                provider=provider,
                criterion=criterion,
                company_profile=company_profile,
                conversation=conversation,
                generated_response=generated_response,
                reference_answer=reference_answer,
                config=config,
            )
            for criterion in criteria
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(score_all_criteria())

    # Filter out exceptions and log them
    criterion_scores: list[CriterionScore] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error("Failed to score criterion %s: %s", criteria[i].name, result)
        else:
            criterion_scores.append(result)

    total_cost = sum(score.total_cost_usd for score in criterion_scores)
    total_api_calls = sum(len(score.raw_scores) for score in criterion_scores)

    # Compute overall score (weighted average of applicable criteria)
    applicable_scores = [s for s in criterion_scores if s.raw_scores]
    if applicable_scores:
        total_weight = sum(
            next(c.weight for c in criteria if c.key == s.criterion_key) for s in applicable_scores
        )
        if total_weight == 0:
            overall_score = 0.0
        else:
            overall_score = (
                sum(
                    s.weighted_score * next(c.weight for c in criteria if c.key == s.criterion_key)
                    for s in applicable_scores
                )
                / total_weight
            )
    else:
        overall_score = 0.0

    elapsed_ms = (time.time() - start_time) * 1000

    return GEvalResult(
        example_id=example.get("original_index", 0),
        task_type=example.get("task", "unknown"),
        criterion_scores=criterion_scores,
        overall_score=overall_score,
        total_cost_usd=total_cost,
        total_api_calls=total_api_calls,
        evaluation_time_ms=elapsed_ms,
    )


def batch_evaluate(
    results: list[dict[str, Any]],
    provider: LiteLLMProvider,
    config: GEvalConfig | None = None,
    criteria: list[EvaluationCriterion] | None = None,
) -> list[GEvalResult]:
    """Evaluate all examples with G-Eval.

    Args:
        results: List of examples from baseline_evaluation.json
        provider: Model provider for judge calls
        config: G-Eval configuration (uses defaults if None)
        criteria: Evaluation criteria (uses CRITERIA if None)

    Returns:
        List of GEvalResult objects
    """
    if config is None:
        config = GEvalConfig()
    if criteria is None:
        criteria = CRITERIA

    logger.info(
        "Starting G-Eval batch evaluation: %d examples, %d criteria, %d samples each",
        len(results),
        len(criteria),
        config.num_samples,
    )
    logger.info("Model: %s", provider.get_model_name())

    geval_results: list[GEvalResult] = []

    for example in tqdm(results, desc="Evaluating examples"):
        example_id = example.get("original_index", "?")
        logger.info("Evaluating example %s (task: %s)", example_id, example.get("task"))

        try:
            result = evaluate_example(provider, example, criteria, config)
            geval_results.append(result)

            logger.info(
                "  Example %s: overall=%.2f, cost=$%.4f",
                example_id,
                result.overall_score,
                result.total_cost_usd,
            )

        except Exception as e:
            logger.error("Failed to evaluate example %s: %s", example_id, e)
            geval_results.append(
                GEvalResult(
                    example_id=example.get("original_index", 0),
                    task_type=example.get("task", "unknown"),
                    criterion_scores=[],
                    overall_score=0.0,
                    total_cost_usd=0.0,
                    total_api_calls=0,
                    evaluation_time_ms=0.0,
                )
            )

    return geval_results


def compute_aggregate_metrics(
    results: list[GEvalResult],
) -> dict[str, Any]:
    """Compute aggregate metrics across all evaluation results.

    Args:
        results: List of GEvalResult objects

    Returns:
        Dictionary with aggregate statistics including overall mean,
        breakdown by task type, and breakdown by criterion
    """
    if not results:
        return {"error": "No results to aggregate"}

    valid_results = [r for r in results if r.criterion_scores]

    # Overall metrics
    overall_scores = [r.overall_score for r in valid_results]
    mean_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    total_cost = sum(r.total_cost_usd for r in results)
    total_api_calls = sum(r.total_api_calls for r in results)

    # By task type
    by_task: dict[str, list[float]] = {}
    for r in valid_results:
        by_task.setdefault(r.task_type, []).append(r.overall_score)

    task_metrics = {
        task: {
            "mean": sum(scores) / len(scores),
            "count": len(scores),
        }
        for task, scores in by_task.items()
    }

    # By criterion
    by_criterion: dict[str, list[float]] = {}
    for r in valid_results:
        for cs in r.criterion_scores:
            if cs.raw_scores:
                by_criterion.setdefault(cs.criterion_key, []).append(cs.weighted_score)

    criterion_metrics = {
        key: {
            "mean": sum(scores) / len(scores),
            "count": len(scores),
        }
        for key, scores in by_criterion.items()
    }

    return {
        "overall_mean": mean_score,
        "total_examples": len(results),
        "valid_examples": len(valid_results),
        "total_cost_usd": total_cost,
        "total_api_calls": total_api_calls,
        "by_task": task_metrics,
        "by_criterion": criterion_metrics,
    }


def results_to_dict(
    results: list[GEvalResult], provider_name: str, config: GEvalConfig
) -> dict[str, Any]:
    """Serialize G-Eval results to a JSON-compatible dictionary.

    Args:
        results: List of GEvalResult objects
        provider_name: Model identifier used for evaluation
        config: G-Eval configuration used

    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "evaluation_metadata": {
            "model": provider_name,
            "num_samples": config.num_samples,
            "temperature": config.temperature,
            "total_examples": len(results),
            "total_api_calls": sum(r.total_api_calls for r in results),
            "total_cost_usd": sum(r.total_cost_usd for r in results),
        },
        "results": [
            {
                "example_id": r.example_id,
                "task_type": r.task_type,
                "overall_score": r.overall_score,
                "total_cost_usd": r.total_cost_usd,
                "total_api_calls": r.total_api_calls,
                "evaluation_time_ms": r.evaluation_time_ms,
                "criterion_scores": [
                    {
                        "criterion_name": cs.criterion_name,
                        "criterion_key": cs.criterion_key,
                        "raw_scores": cs.raw_scores,
                        "score_probabilities": {
                            str(k): v for k, v in cs.score_probabilities.items()
                        },
                        "weighted_score": cs.weighted_score,
                        "justification": cs.justification,
                        "total_cost_usd": cs.total_cost_usd,
                    }
                    for cs in r.criterion_scores
                ],
            }
            for r in results
        ],
        "aggregate_metrics": compute_aggregate_metrics(results),
    }
