"""Tests for G-Eval judge evaluator."""

import pytest

from src.evaluation.judge_evaluator import (
    CRITERIA,
    CriterionScore,
    EvaluationCriterion,
    GEvalConfig,
    GEvalResult,
    compute_aggregate_metrics,
    compute_score_probabilities,
    compute_weighted_score,
    create_geval_prompt,
    evaluate_example,
    parse_company_profile,
    parse_score_from_response,
    results_to_dict,
    score_criterion,
)
from src.evaluation.judge_model_provider import ModelResponse

# --- Fixtures ---

SAMPLE_SYSTEM_MESSAGE = """You are an insurance underwriting co-pilot.

## Tools
Some tools here...

## Company Profile

- Name: Acme Corp
- Annual Revenue: $5,000,000
- Employees: 25
- State: California
- Line of Business: property

## Other Section
Other content here."""

SAMPLE_CRITERION = EvaluationCriterion(
    name="Test Criterion",
    key="test_criterion",
    description="A test criterion for evaluation.",
    evaluation_steps=["Step 1: Check something.", "Step 2: Verify something else."],
)


class MockProvider:
    """Mock model provider for testing."""

    def __init__(self, response_text: str = "Good response. SCORE: 4"):
        self.response_text = response_text
        self.call_count = 0

    def generate(self, _prompt: str, **_kwargs) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(
            content=self.response_text,
            input_tokens=100,
            output_tokens=10,
            model_name="mock-model",
            cost_usd=0.001,
            latency_ms=50.0,
        )

    def get_model_name(self) -> str:
        return "mock-model"


# --- Test: parse_company_profile ---


class TestParseCompanyProfile:
    """Tests for parse_company_profile."""

    def test_extracts_profile_fields(self):
        """Test extraction of all company profile fields."""
        profile = parse_company_profile(SAMPLE_SYSTEM_MESSAGE)
        assert profile["Name"] == "Acme Corp"
        assert profile["Annual Revenue"] == "$5,000,000"
        assert profile["Employees"] == "25"
        assert profile["State"] == "California"
        assert profile["Line of Business"] == "property"

    def test_missing_profile_raises(self):
        """Test ValueError when no company profile section exists."""
        with pytest.raises(ValueError, match="No '## Company Profile'"):
            parse_company_profile("No profile here")

    def test_handles_description_with_commas(self):
        """Test that values with commas are not split incorrectly."""
        msg = """## Company Profile

- Name: Big Corp, Inc.
- Description: Located in Boston, MA"""
        profile = parse_company_profile(msg)
        assert profile["Name"] == "Big Corp, Inc."
        assert profile["Description"] == "Located in Boston, MA"


# --- Test: parse_score_from_response ---


class TestParseScoreFromResponse:
    """Tests for parse_score_from_response."""

    def test_extracts_score(self):
        """Test basic score extraction."""
        assert parse_score_from_response("Analysis here. SCORE: 4") == 4

    def test_extracts_score_zero(self):
        """Test extraction of score 0."""
        assert parse_score_from_response("SCORE: 0") == 0

    def test_extracts_score_five(self):
        """Test extraction of score 5."""
        assert parse_score_from_response("Some text\nSCORE: 5") == 5

    def test_extracts_score_with_extra_whitespace(self):
        """Test extraction with extra whitespace."""
        assert parse_score_from_response("SCORE:  3") == 3

    def test_no_score_raises(self):
        """Test ValueError when no score pattern found."""
        with pytest.raises(ValueError, match="No 'SCORE: X' pattern"):
            parse_score_from_response("No score here")

    def test_score_out_of_range_raises(self):
        """Test ValueError for score > 5."""
        with pytest.raises(ValueError, match="out of valid range"):
            parse_score_from_response("SCORE: 9")


# --- Test: compute_weighted_score ---


class TestComputeWeightedScore:
    """Tests for compute_weighted_score."""

    def test_single_score(self):
        """Test with single score probability."""
        result = compute_weighted_score({4: 1.0})
        assert result == 4.0

    def test_uniform_distribution(self):
        """Test with uniform distribution over 3-5."""
        probs = {3: 1 / 3, 4: 1 / 3, 5: 1 / 3}
        result = compute_weighted_score(probs)
        assert abs(result - 4.0) < 0.01

    def test_weighted_distribution(self):
        """Test with weighted distribution."""
        probs = {3: 0.05, 4: 0.60, 5: 0.35}
        result = compute_weighted_score(probs)
        expected = 3 * 0.05 + 4 * 0.60 + 5 * 0.35  # 4.30
        assert abs(result - expected) < 0.01

    def test_empty_raises(self):
        """Test ValueError on empty dict."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_weighted_score({})


# --- Test: compute_score_probabilities ---


class TestComputeScoreProbabilities:
    """Tests for compute_score_probabilities."""

    def test_uniform_scores(self):
        """Test with identical scores."""
        probs = compute_score_probabilities([4, 4, 4, 4])
        assert probs == {4: 1.0}

    def test_mixed_scores(self):
        """Test with mixed scores."""
        probs = compute_score_probabilities([3, 4, 4, 5])
        assert probs[3] == 0.25
        assert probs[4] == 0.50
        assert probs[5] == 0.25

    def test_empty_raises(self):
        """Test ValueError on empty list."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_score_probabilities([])


# --- Test: GEvalConfig ---


class TestGEvalConfig:
    """Tests for GEvalConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = GEvalConfig()
        assert config.num_samples == 20
        assert config.temperature == 2.0
        assert config.max_tokens == 1024
        assert config.sample_delay == 0.5

    def test_invalid_num_samples_raises(self):
        """Test ValueError for invalid num_samples."""
        with pytest.raises(ValueError, match="num_samples must be >= 1"):
            GEvalConfig(num_samples=0)

    def test_negative_temperature_raises(self):
        """Test ValueError for negative temperature."""
        with pytest.raises(ValueError, match="temperature must be >= 0.0"):
            GEvalConfig(temperature=-1.0)


# --- Test: EvaluationCriterion ---


class TestEvaluationCriterion:
    """Tests for EvaluationCriterion dataclass."""

    def test_creation(self):
        """Test basic creation."""
        criterion = EvaluationCriterion(
            name="Test",
            key="test",
            description="A test criterion",
            evaluation_steps=["Step 1", "Step 2"],
        )
        assert criterion.name == "Test"
        assert criterion.weight == 1.0
        assert criterion.optional is False
        assert len(criterion.evaluation_steps) == 2

    def test_default_criteria_defined(self):
        """Test that CRITERIA list is properly defined."""
        assert len(CRITERIA) == 6
        for c in CRITERIA:
            assert c.name
            assert c.key
            assert c.description
            assert len(c.evaluation_steps) >= 3


# --- Test: create_geval_prompt ---


class TestCreateGEvalPrompt:
    """Tests for create_geval_prompt."""

    def test_contains_criterion_name(self):
        """Test prompt includes criterion name."""
        prompt = create_geval_prompt(
            criterion=SAMPLE_CRITERION,
            company_profile={"Name": "Acme"},
            conversation=[{"role": "user", "content": "Hello"}],
            generated_response="Response here",
            reference_answer="Reference here",
        )
        assert "Test Criterion" in prompt

    def test_contains_evaluation_steps(self):
        """Test prompt includes evaluation steps."""
        prompt = create_geval_prompt(
            criterion=SAMPLE_CRITERION,
            company_profile={"Name": "Acme"},
            conversation=[{"role": "user", "content": "Hello"}],
            generated_response="Response",
            reference_answer="Reference",
        )
        assert "Step 1: Check something." in prompt
        assert "Step 2: Verify something else." in prompt

    def test_contains_company_profile(self):
        """Test prompt includes company profile."""
        prompt = create_geval_prompt(
            criterion=SAMPLE_CRITERION,
            company_profile={"Name": "Acme", "Revenue": "$5M"},
            conversation=[],
            generated_response="Response",
            reference_answer="Reference",
        )
        assert "Name: Acme" in prompt
        assert "Revenue: $5M" in prompt

    def test_ends_with_score_prefix(self):
        """Test prompt ends with SCORE: prefix for model to complete."""
        prompt = create_geval_prompt(
            criterion=SAMPLE_CRITERION,
            company_profile={},
            conversation=[],
            generated_response="Response",
            reference_answer="Reference",
        )
        assert prompt.rstrip().endswith("SCORE:")


# --- Test: score_criterion ---


class TestScoreCriterion:
    """Tests for score_criterion with mock provider."""

    def test_returns_criterion_score(self):
        """Test that scoring returns a valid CriterionScore."""
        provider = MockProvider("Good. SCORE: 4")
        config = GEvalConfig(num_samples=3, sample_delay=0.0)

        result = score_criterion(
            provider=provider,
            criterion=SAMPLE_CRITERION,
            company_profile={"Name": "Acme"},
            conversation=[],
            generated_response="Response",
            reference_answer="Reference",
            config=config,
        )

        assert isinstance(result, CriterionScore)
        assert result.criterion_name == "Test Criterion"
        assert result.criterion_key == "test_criterion"
        assert result.raw_scores == [4, 4, 4]
        assert result.weighted_score == 4.0
        assert result.total_cost_usd == pytest.approx(0.003)

    def test_handles_parse_failures(self):
        """Test graceful handling when score parsing fails."""
        provider = MockProvider("No score in this response")
        config = GEvalConfig(num_samples=3, sample_delay=0.0)

        result = score_criterion(
            provider=provider,
            criterion=SAMPLE_CRITERION,
            company_profile={"Name": "Acme"},
            conversation=[],
            generated_response="Response",
            reference_answer="Reference",
            config=config,
        )

        assert result.raw_scores == []
        assert result.weighted_score == 0.0

    def test_tracks_api_calls(self):
        """Test that provider is called num_samples times."""
        provider = MockProvider("SCORE: 3")
        config = GEvalConfig(num_samples=5, sample_delay=0.0)

        score_criterion(
            provider=provider,
            criterion=SAMPLE_CRITERION,
            company_profile={},
            conversation=[],
            generated_response="R",
            reference_answer="R",
            config=config,
        )

        assert provider.call_count == 5


# --- Test: evaluate_example ---


class TestEvaluateExample:
    """Tests for evaluate_example."""

    def test_evaluates_all_criteria(self):
        """Test that all criteria are evaluated."""
        provider = MockProvider("SCORE: 4")
        config = GEvalConfig(num_samples=2, sample_delay=0.0)
        criteria = [SAMPLE_CRITERION]

        example = {
            "original_index": 1,
            "task": "Policy Limits",
            "messages": [
                {"content": SAMPLE_SYSTEM_MESSAGE, "role": "system"},
                {"content": "Hello", "role": "user"},
            ],
            "generated_response": "Response here",
            "target_response": "Reference here",
        }

        result = evaluate_example(provider, example, criteria, config)

        assert isinstance(result, GEvalResult)
        assert result.example_id == 1
        assert result.task_type == "Policy Limits"
        assert len(result.criterion_scores) == 1
        assert result.overall_score == 4.0
        assert result.total_api_calls == 2


# --- Test: compute_aggregate_metrics ---


class TestComputeAggregateMetrics:
    """Tests for compute_aggregate_metrics."""

    def test_computes_overall_mean(self):
        """Test overall mean computation."""
        results = [
            GEvalResult(
                example_id=1,
                task_type="A",
                criterion_scores=[CriterionScore("c1", "c1", [4], {4: 1.0}, 4.0, "", 0.001)],
                overall_score=4.0,
                total_cost_usd=0.01,
                total_api_calls=20,
                evaluation_time_ms=1000,
            ),
            GEvalResult(
                example_id=2,
                task_type="B",
                criterion_scores=[CriterionScore("c1", "c1", [3], {3: 1.0}, 3.0, "", 0.001)],
                overall_score=3.0,
                total_cost_usd=0.01,
                total_api_calls=20,
                evaluation_time_ms=1000,
            ),
        ]

        metrics = compute_aggregate_metrics(results)
        assert metrics["overall_mean"] == 3.5
        assert metrics["total_examples"] == 2
        assert metrics["total_cost_usd"] == 0.02

    def test_groups_by_task(self):
        """Test grouping by task type."""
        results = [
            GEvalResult(
                1,
                "Policy Limits",
                [CriterionScore("c1", "c1", [4], {4: 1.0}, 4.0, "", 0.0)],
                4.0,
                0.0,
                0,
                0.0,
            ),
            GEvalResult(
                2,
                "Policy Limits",
                [CriterionScore("c1", "c1", [3], {3: 1.0}, 3.0, "", 0.0)],
                3.0,
                0.0,
                0,
                0.0,
            ),
            GEvalResult(
                3,
                "Appetite",
                [CriterionScore("c1", "c1", [5], {5: 1.0}, 5.0, "", 0.0)],
                5.0,
                0.0,
                0,
                0.0,
            ),
        ]

        metrics = compute_aggregate_metrics(results)
        assert metrics["by_task"]["Policy Limits"]["mean"] == 3.5
        assert metrics["by_task"]["Appetite"]["mean"] == 5.0

    def test_empty_results(self):
        """Test with empty results."""
        metrics = compute_aggregate_metrics([])
        assert "error" in metrics


# --- Test: results_to_dict ---


class TestResultsToDict:
    """Tests for results_to_dict serialization."""

    def test_serializes_metadata(self):
        """Test that metadata is included."""
        results = [
            GEvalResult(1, "A", [], 0.0, 0.0, 0, 0.0),
        ]
        config = GEvalConfig(num_samples=10, temperature=1.5)

        output = results_to_dict(results, "test-model", config)

        assert output["evaluation_metadata"]["model"] == "test-model"
        assert output["evaluation_metadata"]["num_samples"] == 10
        assert output["evaluation_metadata"]["temperature"] == 1.5
        assert "results" in output
        assert "aggregate_metrics" in output
