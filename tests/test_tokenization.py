"""Tests for tokenization utilities."""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from src.data.tokenization import (
    compute_token_statistics,
    format_conversation_for_training,
    get_recommended_max_length,
    load_tokenizer,
    tokenize_dataset,
    tokenize_example,
)


@pytest.fixture
def sample_tokenizer():
    """Load a small tokenizer for testing."""
    return load_tokenizer("gpt2")


@pytest.fixture
def sample_example():
    """Create a sample preprocessed example."""
    return {
        "company_profile": {
            "company_name": "Test Corp",
            "annual_revenue": "$1M",
            "number_of_employees": "50",
            "industry": "Tech",
            "state": "CA",
        },
        "conversation": [
            {"role": "user", "content": "What insurance do we need?"},
            {"role": "assistant", "content": "Based on your profile, I recommend..."},
        ],
        "formatted_text": "Company Profile:\n- Name: Test Corp\n...",
        "num_turns": 2,
        "original_index": 0,
    }


def test_load_tokenizer():
    """Test tokenizer loading."""
    tokenizer = load_tokenizer("gpt2")

    assert tokenizer is not None
    assert tokenizer.pad_token is not None
    assert tokenizer.eos_token is not None


def test_load_tokenizer_invalid():
    """Test loading invalid tokenizer."""
    with pytest.raises(ValueError):
        load_tokenizer("nonexistent-model-xyz")


def test_format_conversation_for_training(sample_example, sample_tokenizer):
    """Test conversation formatting."""
    text = format_conversation_for_training(sample_example, sample_tokenizer)

    assert isinstance(text, str)
    assert len(text) > 0
    # Should use formatted_text if available
    assert "Company Profile" in text or "Test Corp" in text


def test_tokenize_example(sample_example, sample_tokenizer):
    """Test single example tokenization."""
    tokenized = tokenize_example(
        sample_example,
        sample_tokenizer,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert "labels" in tokenized

    # Check lengths
    assert len(tokenized["input_ids"]) == 512
    assert len(tokenized["attention_mask"]) == 512
    assert len(tokenized["labels"]) == 512

    # Check metadata
    assert tokenized["original_index"] == 0
    assert tokenized["num_turns"] == 2


def test_tokenize_dataset(sample_tokenizer):
    """Test dataset tokenization."""
    data = {
        "formatted_text": ["Short text", "Another short text"],
        "num_turns": [1, 2],
        "original_index": [0, 1],
    }

    dataset = Dataset.from_dict(data)

    tokenized = tokenize_dataset(
        dataset,
        sample_tokenizer,
        max_length=128,
        verbose=False,
    )

    assert len(tokenized) == 2
    assert "input_ids" in tokenized.column_names
    assert "attention_mask" in tokenized.column_names
    assert "labels" in tokenized.column_names


def test_compute_token_statistics(sample_tokenizer):
    """Test token statistics computation."""
    data = {
        "formatted_text": ["Short text"] * 10,
        "num_turns": [1] * 10,
    }

    dataset = Dataset.from_dict(data)
    stats = compute_token_statistics(dataset, sample_tokenizer)

    assert "total_examples" in stats
    assert stats["total_examples"] == 10
    assert "mean" in stats
    assert "median" in stats
    assert "min" in stats
    assert "max" in stats
    assert "percentiles" in stats


def test_get_recommended_max_length(sample_tokenizer):
    """Test recommended max length calculation."""
    data = {
        "formatted_text": ["Short text"] * 5 + ["Much longer text " * 50] * 5,
        "num_turns": [1] * 10,
    }

    dataset = Dataset.from_dict(data)
    max_length = get_recommended_max_length(dataset, sample_tokenizer, percentile=95)

    assert isinstance(max_length, int)
    assert max_length > 0
