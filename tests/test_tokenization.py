"""Tests for tokenization utilities."""

import pytest
from datasets import Dataset

from src.data.tokenization import (
    compute_token_statistics,
    format_messages_for_tokenization,
    get_recommended_max_length,
    load_tokenizer,
    mark_truncated_examples,
    tokenize_dataset,
    tokenize_example,
)


@pytest.fixture
def sample_tokenizer():
    """Load a small tokenizer for testing."""
    return load_tokenizer("Qwen/Qwen3-0.6B")


@pytest.fixture
def sample_example():
    """Create a sample preprocessed example with messages."""
    return {
        "messages": [
            {"role": "system", "content": "You are an insurance underwriting co-pilot."},
            {"role": "user", "content": "What insurance do we need?"},
            {
                "role": "assistant",
                "content": "Based on your profile, I recommend general liability and property coverage.",
            },
        ],
        "num_turns": 3,
        "original_index": 0,
    }


def test_load_tokenizer():
    """Test tokenizer loading."""
    tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")

    assert tokenizer is not None
    assert tokenizer.pad_token is not None
    assert tokenizer.eos_token is not None


def test_load_tokenizer_invalid():
    """Test loading invalid tokenizer."""
    with pytest.raises(ValueError):
        load_tokenizer("nonexistent-model-xyz")


def test_format_messages_for_tokenization(sample_example, sample_tokenizer):
    """Test messages formatting using chat template."""
    text = format_messages_for_tokenization(sample_example["messages"], sample_tokenizer)

    assert isinstance(text, str)
    assert len(text) > 0
    # Should contain message content
    assert "insurance" in text.lower()


def test_format_messages_for_tokenization_empty(sample_tokenizer):
    """Test that empty messages list raises ValueError."""
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        format_messages_for_tokenization([], sample_tokenizer)


def test_tokenize_example_missing_messages(sample_tokenizer):
    """Test that example without messages field raises ValueError."""
    invalid_example = {
        "num_turns": 1,
        "original_index": 0,
    }

    with pytest.raises(ValueError, match="Example must contain 'messages' field"):
        tokenize_example(invalid_example, sample_tokenizer)


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
    assert tokenized["num_turns"] == 3


def test_tokenize_dataset(sample_tokenizer):
    """Test dataset tokenization."""
    data = {
        "messages": [
            [{"role": "user", "content": "Short text"}],
            [{"role": "user", "content": "Another short text"}],
        ],
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
        "messages": [[{"role": "user", "content": "Short text"}]] * 10,
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
        "messages": (
            [[{"role": "user", "content": "Short text"}]] * 5
            + [[{"role": "user", "content": "Much longer text " * 50}]] * 5
        ),
        "num_turns": [1] * 10,
    }

    dataset = Dataset.from_dict(data)
    max_length = get_recommended_max_length(dataset, sample_tokenizer, percentile=95)

    assert isinstance(max_length, int)
    assert max_length > 0


def test_get_recommended_max_length_invalid_percentile(sample_tokenizer):
    """Test that invalid percentile raises ValueError."""
    data = {
        "messages": [[{"role": "user", "content": "Short text"}]],
        "num_turns": [1],
    }

    dataset = Dataset.from_dict(data)

    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        get_recommended_max_length(dataset, sample_tokenizer, percentile=150)

    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        get_recommended_max_length(dataset, sample_tokenizer, percentile=-10)


def test_mark_truncated_examples(sample_tokenizer):
    """Test marking examples that will be truncated."""
    # Create dataset with short and long examples
    short_text = "This is a short text."
    long_text = "Very long text. " * 100  # ~300 tokens

    data = {
        "messages": [
            [{"role": "user", "content": short_text}],
            [{"role": "user", "content": long_text}],
            [{"role": "user", "content": short_text}],
        ],
        "num_turns": [1, 1, 1],
    }

    dataset = Dataset.from_dict(data)

    # Use a small max_length to trigger truncation on the long example
    result_dataset, stats = mark_truncated_examples(
        dataset, sample_tokenizer, max_length=50, verbose=False
    )

    # Check that metadata columns were added
    assert "token_count" in result_dataset.column_names
    assert "will_truncate" in result_dataset.column_names
    assert "tokens_over_limit" in result_dataset.column_names

    # Check statistics
    assert stats["max_length"] == 50
    assert stats["total_examples"] == 3
    assert stats["truncated_count"] == 1  # Only the long text exceeds 50 tokens
    assert stats["truncated_indices"] == [1]  # Index of long text
    assert stats["max_tokens_over"] > 0

    # Verify the actual data
    assert result_dataset["will_truncate"][0] is False
    assert result_dataset["will_truncate"][1] is True
    assert result_dataset["will_truncate"][2] is False


def test_mark_truncated_examples_empty_dataset(sample_tokenizer):
    """Test that empty dataset raises ValueError."""
    data = {
        "messages": [],
        "num_turns": [],
    }

    dataset = Dataset.from_dict(data)

    with pytest.raises(ValueError, match="Cannot process empty dataset"):
        mark_truncated_examples(dataset, sample_tokenizer, verbose=False)


def test_mark_truncated_examples_no_truncation(sample_tokenizer):
    """Test when no examples need truncation."""
    data = {
        "messages": [
            [{"role": "user", "content": "Short text"}],
            [{"role": "user", "content": "Another short"}],
        ],
        "num_turns": [1, 1],
    }

    dataset = Dataset.from_dict(data)

    # Use a very large max_length
    result_dataset, stats = mark_truncated_examples(
        dataset, sample_tokenizer, max_length=10000, verbose=False
    )

    assert stats["truncated_count"] == 0
    assert stats["truncated_percentage"] == 0.0
    assert stats["truncated_indices"] == []
    assert stats["avg_tokens_over"] == 0
    assert all(not t for t in result_dataset["will_truncate"])
