"""G-Eval based Model-as-Judge evaluation.

Implements the G-Eval framework (https://arxiv.org/abs/2303.16634) for evaluating
insurance underwriting assistant responses. Uses chain-of-thought evaluation steps
and probability-based scoring via multiple sampling for fine-grained, continuous scores.

The evaluator is provider-agnostic through the model_provider abstraction layer,
supporting any LLM provider (Anthropic, OpenAI, etc.) via LiteLLM.

Example:
    from src.evaluation.model_provider import create_provider

    provider = create_provider("claude-3-5-sonnet-20241022")
    config = GEvalConfig(num_samples=20, temperature=2.0)
    results = batch_evaluate(evaluation_data, provider, config)
"""

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from src.evaluation.model_provider import LiteLLMProvider, ModelResponse

logger = logging.getLogger(__name__)


# --- Data Classes ---


@dataclass
class EvaluationCriterion:
    """A single evaluation criterion with G-Eval chain-of-thought steps.

    Attributes:
        name: Human-readable criterion name
        key: Machine-readable key (snake_case)
        description: Full description of what this criterion evaluates
        evaluation_steps: Chain-of-thought steps for systematic evaluation
        weight: Weight for aggregation (default 1.0)
        optional: Whether this criterion may be N/A for some examples
    """

    name: str
    key: str
    description: str
    evaluation_steps: list[str] = field(default_factory=list)
    weight: float = 1.0
    optional: bool = False


@dataclass
class GEvalConfig:
    """Configuration for G-Eval scoring.

    Attributes:
        num_samples: Number of samples for probability estimation
        temperature: Sampling temperature (high for diversity)
        max_tokens: Maximum tokens per evaluation response
        sample_delay: Seconds to wait between samples (rate limiting)
    """

    num_samples: int = 20
    temperature: float = 2.0
    max_tokens: int = 512
    sample_delay: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")


@dataclass
class CriterionScore:
    """Score for a single criterion from G-Eval.

    Attributes:
        criterion_name: Name of the evaluated criterion
        criterion_key: Machine-readable key
        raw_scores: All sampled scores
        score_probabilities: Probability distribution over scores
        weighted_score: Final G-Eval probability-weighted score
        justification: Sample justification from one evaluation
        total_cost_usd: Total API cost for this criterion
    """

    criterion_name: str
    criterion_key: str
    raw_scores: list[int]
    score_probabilities: dict[int, float]
    weighted_score: float
    justification: str
    total_cost_usd: float


@dataclass
class GEvalResult:
    """Complete G-Eval evaluation for one example.

    Attributes:
        example_id: Identifier for the evaluated example
        task_type: Type of underwriting task
        criterion_scores: Scores for each criterion
        overall_score: Weighted average across criteria
        total_cost_usd: Total API cost for this example
        total_api_calls: Number of API calls made
        evaluation_time_ms: Total evaluation time in milliseconds
    """

    example_id: int
    task_type: str
    criterion_scores: list[CriterionScore]
    overall_score: float
    total_cost_usd: float
    total_api_calls: int
    evaluation_time_ms: float


# --- Evaluation Criteria ---

CRITERIA = [
    EvaluationCriterion(
        name="Appetite Decision Accuracy",
        key="appetite_decision_accuracy",
        description=(
            "Does the response correctly determine if the company is in/out/qualified "
            "for appetite? Are the reasons aligned with the underwriting guidelines?"
        ),
        evaluation_steps=[
            "Check if the response explicitly states an appetite decision (in-appetite, out-of-appetite, or qualified).",
            "Verify the decision matches what the underwriting guidelines specify for this company's NAICS code and line of business.",
            "Assess whether the reasoning references specific guideline criteria (e.g., small business qualification, building construction type, vehicle count, payroll threshold).",
            "Compare the decision and reasoning against the reference answer.",
        ],
        optional=True,
    ),
    EvaluationCriterion(
        name="Product Recommendation Relevance",
        key="product_recommendation_relevance",
        description=(
            "Are the recommended insurance products appropriate for the company profile? "
            "Are important coverage types identified?"
        ),
        evaluation_steps=[
            "Identify what insurance products or coverage types are recommended in the response.",
            "Check if the recommended products are relevant to the company's industry, size, and risk profile.",
            "Verify no critical coverage types are missing given the company's business description.",
            "Compare recommendations against the reference answer for completeness.",
        ],
        optional=True,
    ),
    EvaluationCriterion(
        name="Limit/Deductible Accuracy",
        key="limit_deductible_accuracy",
        description=(
            "Are suggested policy limits and deductibles appropriate? "
            "Do they align with the underwriting guidelines and company size?"
        ),
        evaluation_steps=[
            "Identify the specific dollar amounts for limits and deductibles mentioned in the response.",
            "Check if these amounts match the underwriting guidelines for the company's NAICS code and line of business.",
            "Verify the response distinguishes between per-occurrence and aggregate limits where applicable.",
            "Compare the suggested amounts against the reference answer.",
        ],
        optional=True,
    ),
    EvaluationCriterion(
        name="Risk Assessment Correctness",
        key="risk_assessment_correctness",
        description=(
            "Are risk factors appropriately identified? "
            "Is the assessment aligned with insurance underwriting principles?"
        ),
        evaluation_steps=[
            "Identify the risk factors mentioned in the response.",
            "Check if the identified risks are relevant to the company's industry and operations.",
            "Assess whether any material risks are missing from the evaluation.",
            "Verify the overall risk assessment is consistent with the identified factors.",
        ],
    ),
    EvaluationCriterion(
        name="Use of Company Profile",
        key="use_of_company_profile",
        description=(
            "Does the response reference specific company details (revenue, employees, "
            "industry, etc.)? Are recommendations tailored to the company's situation?"
        ),
        evaluation_steps=[
            "Check if the response references specific company attributes (name, revenue, employees, location, industry).",
            "Assess whether recommendations are tailored to the company's specific situation rather than generic.",
            "Verify the response uses the correct company details (no hallucinated or incorrect data).",
            "Compare the level of personalization against the reference answer.",
        ],
    ),
    EvaluationCriterion(
        name="Multi-Turn Coherence",
        key="multi_turn_coherence",
        description=(
            "Does the response maintain context from previous conversation turns? "
            "Are follow-up questions answered appropriately?"
        ),
        evaluation_steps=[
            "Review the conversation history and identify what context should be maintained.",
            "Check if the response addresses the specific question or request from the latest turn.",
            "Verify the response is consistent with information provided in earlier turns.",
            "Assess whether the response builds on prior exchanges rather than repeating or contradicting them.",
        ],
    ),
]


# --- Utility Functions ---


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
        raise ValueError(f"No 'SCORE: X' pattern found in response: {response_text[-100:]}")

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


# --- G-Eval Prompt ---


def create_geval_prompt(
    criterion: EvaluationCriterion,
    company_profile: dict[str, str],
    conversation: list[dict[str, str]],
    generated_response: str,
    reference_answer: str,
) -> str:
    """Create a G-Eval structured evaluation prompt for a single criterion.

    Args:
        criterion: The evaluation criterion with chain-of-thought steps
        company_profile: Parsed company profile dictionary
        conversation: Conversation history (excluding system message)
        generated_response: The model's response to evaluate
        reference_answer: The ground truth reference answer

    Returns:
        Formatted evaluation prompt string
    """
    steps_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(criterion.evaluation_steps))

    profile_str = "\n".join(f"- {k}: {v}" for k, v in company_profile.items())

    conv_str = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation)

    return f"""You are an expert insurance underwriting evaluator. Your task is to evaluate an AI assistant's response on the following criterion.

## Criterion: {criterion.name}
{criterion.description}

## Evaluation Steps
Follow these steps systematically:
{steps_str}

## Company Profile
{profile_str}

## Conversation History
{conv_str}

## Generated Response (to evaluate)
{generated_response}

## Reference Answer
{reference_answer}

## Scoring Guide
- 5: Excellent, fully correct and complete
- 4: Good, minor issues only
- 3: Adequate, some errors or omissions
- 2: Poor, significant errors
- 1: Very poor, mostly incorrect
- 0: Completely wrong or not addressed

## Instructions
1. Work through each evaluation step above.
2. Compare the generated response against the reference answer.
3. Provide a brief justification for your score.
4. End your response with exactly: SCORE: <0-5>

SCORE: """


# --- Core G-Eval Scoring ---


def score_criterion(
    provider: LiteLLMProvider,
    criterion: EvaluationCriterion,
    company_profile: dict[str, str],
    conversation: list[dict[str, str]],
    generated_response: str,
    reference_answer: str,
    config: GEvalConfig,
) -> CriterionScore:
    """Score a single criterion using G-Eval multiple sampling.

    Samples the judge model N times with high temperature and aggregates
    scores into a probability-weighted continuous score.

    Args:
        provider: Model provider for judge calls
        criterion: Criterion to evaluate
        company_profile: Parsed company profile
        conversation: Conversation history
        generated_response: Model response to evaluate
        reference_answer: Ground truth reference
        config: G-Eval configuration

    Returns:
        CriterionScore with probability-weighted score and metadata
    """
    prompt = create_geval_prompt(
        criterion=criterion,
        company_profile=company_profile,
        conversation=conversation,
        generated_response=generated_response,
        reference_answer=reference_answer,
    )

    raw_scores: list[int] = []
    total_cost = 0.0
    justification = ""

    for i in range(config.num_samples):
        try:
            response: ModelResponse = provider.generate(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            total_cost += response.cost_usd

            score = parse_score_from_response(response.content)
            raw_scores.append(score)

            # Capture first justification for logging
            if not justification:
                justification = response.content

        except ValueError as e:
            logger.warning(
                "Failed to extract score for %s (sample %d/%d): %s",
                criterion.name,
                i + 1,
                config.num_samples,
                e,
            )

        if i < config.num_samples - 1:
            time.sleep(config.sample_delay)

    if not raw_scores:
        logger.error("No valid scores collected for criterion: %s", criterion.name)
        return CriterionScore(
            criterion_name=criterion.name,
            criterion_key=criterion.key,
            raw_scores=[],
            score_probabilities={},
            weighted_score=0.0,
            justification="No valid scores collected",
            total_cost_usd=total_cost,
        )

    score_probs = compute_score_probabilities(raw_scores)
    weighted = compute_weighted_score(score_probs)

    return CriterionScore(
        criterion_name=criterion.name,
        criterion_key=criterion.key,
        raw_scores=raw_scores,
        score_probabilities=score_probs,
        weighted_score=weighted,
        justification=justification,
        total_cost_usd=total_cost,
    )


# --- Example-Level Evaluation ---


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

    # Score each criterion
    criterion_scores: list[CriterionScore] = []
    total_cost = 0.0
    total_api_calls = 0

    for criterion in criteria:
        logger.info("  Scoring criterion: %s", criterion.name)

        score = score_criterion(
            provider=provider,
            criterion=criterion,
            company_profile=company_profile,
            conversation=conversation,
            generated_response=generated_response,
            reference_answer=reference_answer,
            config=config,
        )

        criterion_scores.append(score)
        total_cost += score.total_cost_usd
        total_api_calls += len(score.raw_scores)

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


# --- Batch Evaluation ---


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


# --- Aggregate Metrics ---


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


# --- Serialization ---


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
