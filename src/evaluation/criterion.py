"""Evaluation criteria definitions for G-Eval."""

from dataclasses import dataclass, field


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
