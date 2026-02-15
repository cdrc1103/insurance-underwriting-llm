"""Data models for G-Eval evaluation."""

from dataclasses import dataclass
from typing import Any


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
    max_tokens: int = 1024
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary with string keys for score_probabilities
        """
        return {
            "criterion_name": self.criterion_name,
            "criterion_key": self.criterion_key,
            "raw_scores": self.raw_scores,
            "score_probabilities": {str(k): v for k, v in self.score_probabilities.items()},
            "weighted_score": self.weighted_score,
            "justification": self.justification,
            "total_cost_usd": self.total_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionScore":
        """Create from dictionary with string keys.

        Args:
            data: Dictionary from JSON deserialization

        Returns:
            CriterionScore instance
        """
        return cls(
            criterion_name=data["criterion_name"],
            criterion_key=data["criterion_key"],
            raw_scores=data["raw_scores"],
            score_probabilities={int(k): v for k, v in data["score_probabilities"].items()},
            weighted_score=data["weighted_score"],
            justification=data["justification"],
            total_cost_usd=data["total_cost_usd"],
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary ready for JSON serialization
        """
        return {
            "example_id": self.example_id,
            "task_type": self.task_type,
            "overall_score": self.overall_score,
            "total_cost_usd": self.total_cost_usd,
            "total_api_calls": self.total_api_calls,
            "evaluation_time_ms": self.evaluation_time_ms,
            "criterion_scores": [cs.to_dict() for cs in self.criterion_scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GEvalResult":
        """Create from dictionary.

        Args:
            data: Dictionary from JSON deserialization

        Returns:
            GEvalResult instance
        """
        return cls(
            example_id=data["example_id"],
            task_type=data["task_type"],
            overall_score=data["overall_score"],
            total_cost_usd=data["total_cost_usd"],
            total_api_calls=data["total_api_calls"],
            evaluation_time_ms=data["evaluation_time_ms"],
            criterion_scores=[CriterionScore.from_dict(cs) for cs in data["criterion_scores"]],
        )
