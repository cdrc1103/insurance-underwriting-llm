"""Tests for data loading utilities."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset, DatasetDict

from src.data.loader import (
    get_dataset_statistics,
    load_dataset_split,
    load_insurance_dataset,
    save_dataset_split,
)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    data = {
        "text": ["example 1", "example 2", "example 3"],
        "label": [0, 1, 0],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def mock_dataset_dict():
    """Create a mock dataset dict for testing."""
    data = {
        "text": ["example 1", "example 2", "example 3"],
        "label": [0, 1, 0],
    }
    dataset = Dataset.from_dict(data)
    return DatasetDict({"train": dataset})


def test_get_dataset_statistics(mock_dataset):
    """Test dataset statistics computation."""
    stats = get_dataset_statistics(mock_dataset)

    assert "num_examples" in stats
    assert stats["num_examples"] == 3
    assert "features" in stats
    assert "column_names" in stats
    assert "example_keys" in stats
    assert set(stats["column_names"]) == {"text", "label"}


def test_get_dataset_statistics_empty():
    """Test statistics for empty dataset."""
    empty_dataset = Dataset.from_dict({"text": [], "label": []})
    stats = get_dataset_statistics(empty_dataset)

    assert stats["num_examples"] == 0
    assert "features" in stats


def test_save_and_load_dataset_split(mock_dataset, tmp_path):
    """Test saving and loading dataset splits."""
    # Save dataset
    save_dataset_split(mock_dataset, tmp_path, split_name="test")

    # Verify file was created
    assert (tmp_path / "test").exists()

    # Load dataset
    loaded_dataset = load_dataset_split(tmp_path, split_name="test")

    # Verify loaded dataset matches original
    assert len(loaded_dataset) == len(mock_dataset)
    assert loaded_dataset.column_names == mock_dataset.column_names


def test_load_dataset_split_not_found(tmp_path):
    """Test loading non-existent dataset split."""
    with pytest.raises(FileNotFoundError):
        load_dataset_split(tmp_path, split_name="nonexistent")


@patch("src.data.loader.load_dataset")
def test_load_insurance_dataset_download(mock_load_dataset, mock_dataset_dict):
    """Test loading dataset from Hugging Face."""
    mock_load_dataset.return_value = mock_dataset_dict

    dataset = load_insurance_dataset(
        dataset_name="test/dataset",
        force_download=True,
    )

    assert isinstance(dataset, DatasetDict)
    mock_load_dataset.assert_called_once_with("test/dataset")


@patch("src.data.loader.load_from_disk")
def test_load_insurance_dataset_from_cache(mock_load_from_disk, mock_dataset_dict, tmp_path):
    """Test loading dataset from cache."""
    # Create fake cache directory
    cache_path = tmp_path / "raw" / "insurance_underwriting"
    cache_path.mkdir(parents=True)

    mock_load_from_disk.return_value = mock_dataset_dict

    dataset = load_insurance_dataset(
        cache_dir=tmp_path,
        force_download=False,
    )

    assert isinstance(dataset, DatasetDict)
    mock_load_from_disk.assert_called_once()


@patch("src.data.loader.load_dataset")
def test_load_insurance_dataset_error(mock_load_dataset):
    """Test error handling when dataset cannot be loaded."""
    mock_load_dataset.side_effect = Exception("Network error")

    with pytest.raises(ValueError, match="Failed to load dataset"):
        load_insurance_dataset(dataset_name="test/dataset")
