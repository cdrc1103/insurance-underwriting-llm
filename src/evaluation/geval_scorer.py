"""Core G-Eval scoring logic (async only)."""

import asyncio
import logging

from src.evaluation.criterion import EvaluationCriterion
from src.evaluation.judge_model_provider import LiteLLMProvider, ModelResponse
from src.evaluation.models import CriterionScore, GEvalConfig
from src.evaluation.utils import (
    compute_score_probabilities,
    compute_weighted_score,
    parse_score_from_response,
)

logger = logging.getLogger(__name__)


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


async def score_criterion(
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
            # Log prompt length on first sample for debugging
            if i == 0:
                logger.debug(
                    "Prompt length for %s: %d chars, ~%d tokens",
                    criterion.name,
                    len(prompt),
                    len(prompt) // 4,
                )

            response: ModelResponse = await provider.generate(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            total_cost += response.cost_usd

            # Log response info for debugging
            logger.debug(
                "Response for %s (sample %d): length=%d, tokens_in=%d, tokens_out=%d",
                criterion.name,
                i + 1,
                len(response.content),
                response.input_tokens,
                response.output_tokens,
            )

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
            await asyncio.sleep(config.sample_delay)

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
