"""Model-as-Judge evaluation using Claude API."""

from typing import Any

# Note: Install anthropic SDK with: uv add anthropic
# from anthropic import Anthropic


EVALUATION_RUBRIC = """
You are evaluating an AI insurance underwriting assistant's response.

Evaluate the response on the following criteria (score 0-5 for each):

1. **Appetite Decision Accuracy** (if applicable):
   - Does the response correctly determine if the company is in/out/qualified for appetite?
   - Are the reasons aligned with the underwriting guidelines?

2. **Product Recommendation Relevance** (if applicable):
   - Are the recommended insurance products appropriate for the company profile?
   - Are important coverage types identified?

3. **Limit/Deductible Accuracy** (if applicable):
   - Are suggested policy limits and deductibles appropriate?
   - Do they align with industry standards and company size?

4. **Risk Assessment Correctness**:
   - Are risk factors appropriately identified?
   - Is the assessment aligned with insurance underwriting principles?

5. **Use of Company Profile**:
   - Does the response reference specific company details (revenue, employees, industry, etc.)?
   - Are recommendations tailored to the company's situation?

6. **Multi-Turn Coherence**:
   - Does the response maintain context from previous conversation turns?
   - Are follow-up questions answered appropriately?

Scoring Guide:
- 5: Excellent, fully correct
- 4: Good, minor issues
- 3: Adequate, some errors
- 2: Poor, significant errors
- 1: Very poor, mostly incorrect
- 0: Completely wrong or not applicable

Provide scores as JSON with brief justifications.
"""


def create_judge_prompt(
    company_profile: dict[str, Any],
    conversation: list[dict[str, str]],
    generated_response: str,
    reference_answer: str,
) -> str:
    """
    Create evaluation prompt for Claude judge.

    Args:
        company_profile: Company information dict
        conversation: Conversation history
        generated_response: Model's generated response
        reference_answer: Expected/reference answer

    Returns:
        Formatted judge prompt
    """
    # Format company profile
    profile_str = "\\n".join([f"- {k}: {v}" for k, v in company_profile.items()])

    # Format conversation
    conv_str = "\\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation])

    prompt = f"""
{EVALUATION_RUBRIC}

## Company Profile
{profile_str}

## Conversation History
{conv_str}

## Generated Response (to evaluate)
{generated_response}

## Reference Answer
{reference_answer}

Please evaluate the generated response and provide scores in the following JSON format:
{{
  "appetite_decision_accuracy": {{
    "score": <0-5 or "N/A">,
    "justification": "<brief explanation>"
  }},
  "product_recommendation_relevance": {{
    "score": <0-5 or "N/A">,
    "justification": "<brief explanation>"
  }},
  "limit_deductible_accuracy": {{
    "score": <0-5 or "N/A">,
    "justification": "<brief explanation>"
  }},
  "risk_assessment_correctness": {{
    "score": <0-5>,
    "justification": "<brief explanation>"
  }},
  "use_of_company_profile": {{
    "score": <0-5>,
    "justification": "<brief explanation>"
  }},
  "multi_turn_coherence": {{
    "score": <0-5>,
    "justification": "<brief explanation>"
  }},
  "overall_quality": {{
    "score": <0-5>,
    "justification": "<overall assessment>"
  }}
}}
"""

    return prompt


def evaluate_with_claude(
    result: dict[str, Any],
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
) -> dict[str, Any]:
    """
    Evaluate a single result using Claude as judge.

    Args:
        result: Dict with generated_response, reference_answer, etc.
        api_key: Anthropic API key
        model: Claude model to use for judging

    Returns:
        Dict with evaluation scores and justifications

    Raises:
        ValueError: If evaluation fails

    Note:
        Requires anthropic SDK: uv add anthropic
    """
    # TODO: Implement after anthropic SDK is installed
    # client = Anthropic(api_key=api_key)
    #
    # # Extract company profile from messages (system message)
    # system_msg = result["messages"][0]["content"]
    # # Parse company profile from system message...
    #
    # prompt = create_judge_prompt(
    #     company_profile={},  # Parsed from system message
    #     conversation=result["messages"][1:],  # Skip system
    #     generated_response=result["generated_response"],
    #     reference_answer=result["reference_answer"],
    # )
    #
    # response = client.messages.create(
    #     model=model,
    #     max_tokens=2048,
    #     temperature=0,  # Deterministic evaluation
    #     messages=[{"role": "user", "content": prompt}],
    # )
    #
    # # Parse JSON response
    # import json
    # evaluation = json.loads(response.content[0].text)
    # return evaluation

    raise NotImplementedError("Install anthropic SDK first: uv add anthropic")


def batch_evaluate(
    results: list[dict[str, Any]],
    api_key: str,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Evaluate multiple results using Claude as judge.

    Args:
        results: List of result dicts from evaluate_dataset
        api_key: Anthropic API key
        verbose: Whether to print progress

    Returns:
        List of results with evaluation scores added

    Raises:
        ValueError: If evaluation fails
    """
    evaluated_results = []

    for i, result in enumerate(results):
        if verbose and i % 10 == 0:
            print(f"Evaluating example {i}/{len(results)}...")

        try:
            evaluation = evaluate_with_claude(result, api_key)
            result["evaluation"] = evaluation
            evaluated_results.append(result)

        except Exception as e:
            print(f"Failed to evaluate example {i}: {e}")
            result["evaluation"] = {"error": str(e)}
            evaluated_results.append(result)

    return evaluated_results
