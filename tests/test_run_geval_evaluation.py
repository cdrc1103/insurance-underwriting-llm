"""Tests for run_geval_evaluation.py script."""

import json
from pathlib import Path

import pytest

from src.evaluation.models import CriterionScore, GEvalConfig, GEvalResult


@pytest.fixture
def temp_output_file(tmp_path: Path) -> Path:
    """Create a temporary output file path."""
    return tmp_path / "test_results.json"


@pytest.fixture
def sample_geval_results() -> list[GEvalResult]:
    """Create sample GEvalResult objects for testing."""
    return [
        GEvalResult(
            example_id=1,
            task_type="appetite_check",
            criterion_scores=[
                CriterionScore(
                    criterion_name="Test Criterion",
                    criterion_key="test_criterion",
                    raw_scores=[4, 5, 4],
                    score_probabilities={4: 0.67, 5: 0.33},
                    weighted_score=4.33,
                    justification="Test justification",
                    total_cost_usd=0.01,
                )
            ],
            overall_score=4.33,
            total_cost_usd=0.01,
            total_api_calls=3,
            evaluation_time_ms=100.0,
        ),
        GEvalResult(
            example_id=2,
            task_type="product_recommendation",
            criterion_scores=[
                CriterionScore(
                    criterion_name="Test Criterion",
                    criterion_key="test_criterion",
                    raw_scores=[3, 3, 4],
                    score_probabilities={3: 0.67, 4: 0.33},
                    weighted_score=3.33,
                    justification="Another justification",
                    total_cost_usd=0.01,
                )
            ],
            overall_score=3.33,
            total_cost_usd=0.01,
            total_api_calls=3,
            evaluation_time_ms=150.0,
        ),
    ]


def test_save_results_incrementally(
    temp_output_file: Path, sample_geval_results: list[GEvalResult]
) -> None:
    """Test saving results incrementally creates valid JSON."""
    from src.evaluation.utils import save_results_incrementally

    config = GEvalConfig(num_samples=5, temperature=1.0, max_tokens=1024)

    save_results_incrementally(temp_output_file, sample_geval_results, "claude-test", config)

    # Verify file was created
    assert temp_output_file.exists()

    # Verify file contains valid JSON
    with open(temp_output_file) as f:
        data = json.load(f)

    # Check structure
    assert "evaluation_metadata" in data
    assert "results" in data
    assert "aggregate_metrics" in data

    # Check metadata
    assert data["evaluation_metadata"]["model"] == "claude-test"
    assert data["evaluation_metadata"]["num_samples"] == 5
    assert data["evaluation_metadata"]["total_examples"] == 2

    # Check results
    assert len(data["results"]) == 2
    assert data["results"][0]["example_id"] == 1
    assert data["results"][1]["example_id"] == 2


def test_load_existing_results(
    temp_output_file: Path, sample_geval_results: list[GEvalResult]
) -> None:
    """Test loading existing results from file."""
    from src.evaluation.utils import load_existing_results, save_results_incrementally

    config = GEvalConfig(num_samples=5, temperature=1.0, max_tokens=1024)

    # First save some results
    save_results_incrementally(temp_output_file, sample_geval_results, "claude-test", config)

    # Then load them back
    loaded_results, evaluated_ids = load_existing_results(temp_output_file)

    # Verify loaded results match
    assert len(loaded_results) == 2
    assert evaluated_ids == {1, 2}
    assert loaded_results[0].example_id == 1
    assert loaded_results[1].example_id == 2
    assert loaded_results[0].overall_score == 4.33
    assert loaded_results[1].overall_score == 3.33


def test_load_nonexistent_file(temp_output_file: Path) -> None:
    """Test loading from nonexistent file returns empty results."""
    from src.evaluation.utils import load_existing_results

    loaded_results, evaluated_ids = load_existing_results(temp_output_file)

    assert loaded_results == []
    assert evaluated_ids == set()


def test_save_overwrites_existing_file(
    temp_output_file: Path, sample_geval_results: list[GEvalResult]
) -> None:
    """Test saving results overwrites existing file."""
    from src.evaluation.utils import save_results_incrementally

    config = GEvalConfig(num_samples=5, temperature=1.0, max_tokens=1024)

    # Save first version
    save_results_incrementally(temp_output_file, sample_geval_results[:1], "claude-test", config)

    # Verify first version
    with open(temp_output_file) as f:
        data = json.load(f)
    assert len(data["results"]) == 1

    # Save second version with more results
    save_results_incrementally(temp_output_file, sample_geval_results, "claude-test", config)

    # Verify second version overwrote the first
    with open(temp_output_file) as f:
        data = json.load(f)
    assert len(data["results"]) == 2


def test_atomic_write_with_temp_file(
    tmp_path: Path, sample_geval_results: list[GEvalResult]
) -> None:
    """Test that save uses atomic write with temp file."""
    from src.evaluation.utils import save_results_incrementally

    output_file = tmp_path / "test_results.json"
    config = GEvalConfig(num_samples=5, temperature=1.0, max_tokens=1024)

    save_results_incrementally(output_file, sample_geval_results, "claude-test", config)

    # Verify no temp file remains
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    assert not temp_file.exists()

    # Verify output file exists
    assert output_file.exists()
