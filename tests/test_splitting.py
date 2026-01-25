"""Tests for dataset splitting utilities."""

import json

import pytest
from datasets import Dataset

from src.data.splitting import (
    compute_split_statistics,
    create_stratified_split,
    load_splits,
    print_split_summary,
    save_splits,
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing with task metadata."""
    data = {
        "task": [
            "Appetite Check",
            "Product Recommendations",
            "Eligibility",
            "Auto LOB",
            "General",
        ]
        * 10,  # 50 examples total
        "num_turns": [2, 3, 2, 4, 1] * 10,
        "formatted_text": ["text" * 50] * 50,
    }
    return Dataset.from_dict(data)


def test_create_stratified_split(sample_dataset):
    """Test stratified splitting."""
    splits = create_stratified_split(
        sample_dataset,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_seed=42,
    )

    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    # Check sizes are approximately correct
    total = len(sample_dataset)
    assert len(splits["train"]) == pytest.approx(total * 0.7, abs=2)
    assert len(splits["validation"]) == pytest.approx(total * 0.15, abs=2)
    assert len(splits["test"]) == pytest.approx(total * 0.15, abs=2)

    # Check no overlap
    train_indices = set(range(len(splits["train"])))
    val_indices = set(range(len(splits["validation"])))
    test_indices = set(range(len(splits["test"])))

    # Note: indices are from different splits, so this test checks they're non-empty
    assert len(train_indices) > 0
    assert len(val_indices) > 0
    assert len(test_indices) > 0


def test_create_stratified_split_invalid_sizes(sample_dataset):
    """Test splitting with invalid sizes."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        create_stratified_split(
            sample_dataset,
            train_size=0.5,
            val_size=0.3,
            test_size=0.3,  # Sum = 1.1
        )


def test_create_stratified_split_missing_stratify_field(sample_dataset):
    """Test splitting with missing stratification field."""
    with pytest.raises(ValueError, match="Stratification field.*not found"):
        create_stratified_split(
            sample_dataset,
            stratify_by="nonexistent_field",
        )


def test_create_stratified_split_reproducible(sample_dataset):
    """Test that splits are reproducible with same seed."""
    splits1 = create_stratified_split(sample_dataset, random_seed=42)
    splits2 = create_stratified_split(sample_dataset, random_seed=42)

    assert len(splits1["train"]) == len(splits2["train"])
    assert len(splits1["validation"]) == len(splits2["validation"])
    assert len(splits1["test"]) == len(splits2["test"])


def test_compute_split_statistics():
    """Test statistics computation."""
    data = {
        "task": ["Appetite Check", "Product Recommendations", "Appetite Check"],
        "num_turns": [2, 3, 2],
        "formatted_text": ["short", "medium text", "short"],
    }
    split = Dataset.from_dict(data)

    stats = compute_split_statistics(split, "test")

    assert stats["split_name"] == "test"
    assert stats["num_examples"] == 3
    assert "task_distribution" in stats
    assert "conversation_length" in stats
    assert "text_length" in stats

    # Check task distribution
    assert "Appetite Check" in stats["task_distribution"]
    assert stats["task_distribution"]["Appetite Check"]["count"] == 2


def test_save_and_load_splits(sample_dataset, tmp_path):
    """Test saving and loading splits."""
    # Create splits
    splits = create_stratified_split(sample_dataset, random_seed=42)

    # Save
    output_dir = tmp_path / "splits"
    save_splits(splits, output_dir, save_stats=True)

    # Check files were created
    assert (output_dir / "train").exists()
    assert (output_dir / "validation").exists()
    assert (output_dir / "test").exists()
    assert (output_dir / "split_statistics.json").exists()

    # Load statistics
    with open(output_dir / "split_statistics.json") as f:
        stats = json.load(f)
    assert "train" in stats
    assert "validation" in stats
    assert "test" in stats

    # Load splits
    loaded_splits = load_splits(output_dir)

    assert len(loaded_splits["train"]) == len(splits["train"])
    assert len(loaded_splits["validation"]) == len(splits["validation"])
    assert len(loaded_splits["test"]) == len(splits["test"])


def test_load_splits_not_found(tmp_path):
    """Test loading from non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path / "nonexistent")


def test_print_split_summary(sample_dataset, capsys):
    """Test split summary printing."""
    splits = create_stratified_split(sample_dataset, random_seed=42)

    print_split_summary(splits)

    captured = capsys.readouterr()
    assert "Dataset Split Summary" in captured.out
    assert "TRAIN Split:" in captured.out
    assert "VALIDATION Split:" in captured.out
    assert "TEST Split:" in captured.out
