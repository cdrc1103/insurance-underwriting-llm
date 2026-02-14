"""Tests for SFT data preparation with label masking."""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from src.training.data import (
    prepare_conversation_for_sft,
    prepare_sft_dataset,
    tokenize_for_sft,
)


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer for testing."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def sample_example():
    """Create a sample preprocessed example."""
    return {
        "messages": [
            {"role": "system", "content": "You are an insurance underwriting assistant."},
            {"role": "user", "content": "What products are available for a small business?"},
        ],
        "target_response": "Based on the company profile, I recommend general liability insurance.",
        "original_index": 0,
        "num_turns": 1,
    }


@pytest.fixture
def sample_dataset(sample_example):
    """Create a small dataset for testing."""
    examples = [sample_example] * 3
    return Dataset.from_list(examples)


class TestPrepareConversationForSft:
    """Tests for prepare_conversation_for_sft."""

    def test_appends_target_as_assistant(self):
        """Test that target response is added as final assistant message."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        result = prepare_conversation_for_sft(messages, "Hi there!")

        assert len(result) == 3
        assert result[-1]["role"] == "assistant"
        assert result[-1]["content"] == "Hi there!"

    def test_preserves_original_messages(self):
        """Test that original messages are not modified."""
        messages = [{"role": "user", "content": "Hello"}]
        original_len = len(messages)
        prepare_conversation_for_sft(messages, "Response")
        assert len(messages) == original_len

    def test_empty_messages_raises(self):
        """Test that empty messages raises ValueError."""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            prepare_conversation_for_sft([], "Response")

    def test_empty_target_raises(self):
        """Test that empty target raises ValueError."""
        with pytest.raises(ValueError, match="Target response cannot be empty"):
            prepare_conversation_for_sft([{"role": "user", "content": "Hi"}], "")


class TestTokenizeForSft:
    """Tests for tokenize_for_sft."""

    def test_returns_required_keys(self, tokenizer, sample_example):
        """Test that output has input_ids, attention_mask, labels."""
        result = tokenize_for_sft(sample_example, tokenizer, max_length=512)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_labels_have_masked_prompt(self, tokenizer, sample_example):
        """Test that prompt tokens are masked with -100 in labels."""
        result = tokenize_for_sft(sample_example, tokenizer, max_length=512)

        labels = result["labels"]
        # First tokens should be -100 (masked prompt)
        assert labels[0] == -100
        # There should be some non-(-100) tokens (the response)
        non_masked = [token_id for token_id in labels if token_id != -100]
        assert len(non_masked) > 0

    def test_padding_tokens_masked(self, tokenizer, sample_example):
        """Test that padding tokens are masked in labels."""
        result = tokenize_for_sft(sample_example, tokenizer, max_length=512)

        for i, (mask, label) in enumerate(
            zip(result["attention_mask"], result["labels"], strict=True)
        ):
            if mask == 0:  # Padding position
                assert label == -100, f"Padding at position {i} should have label -100"

    def test_preserves_metadata(self, tokenizer, sample_example):
        """Test that metadata fields are preserved."""
        result = tokenize_for_sft(sample_example, tokenizer, max_length=512)
        assert result["original_index"] == 0
        assert result["num_turns"] == 1

    def test_respects_max_length(self, tokenizer, sample_example):
        """Test that output respects max_length."""
        max_length = 128
        result = tokenize_for_sft(sample_example, tokenizer, max_length=max_length)
        assert len(result["input_ids"]) == max_length
        assert len(result["attention_mask"]) == max_length
        assert len(result["labels"]) == max_length

    def test_missing_messages_raises(self, tokenizer):
        """Test that missing messages field raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'messages'"):
            tokenize_for_sft({"target_response": "test"}, tokenizer)

    def test_missing_target_raises(self, tokenizer):
        """Test that missing target_response field raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'target_response'"):
            tokenize_for_sft({"messages": []}, tokenizer)


class TestPrepareSftDataset:
    """Tests for prepare_sft_dataset."""

    def test_tokenizes_all_examples(self, tokenizer, sample_dataset):
        """Test that all examples are tokenized."""
        result = prepare_sft_dataset(sample_dataset, tokenizer, max_length=512)
        assert len(result) == len(sample_dataset)

    def test_output_has_model_columns(self, tokenizer, sample_dataset):
        """Test that output has input_ids, attention_mask, labels."""
        result = prepare_sft_dataset(sample_dataset, tokenizer, max_length=512)
        assert "input_ids" in result.column_names
        assert "attention_mask" in result.column_names
        assert "labels" in result.column_names

    def test_missing_column_raises(self, tokenizer):
        """Test that missing required columns raises ValueError."""
        bad_dataset = Dataset.from_dict({"text": ["hello"]})
        with pytest.raises(ValueError, match="must have 'messages' column"):
            prepare_sft_dataset(bad_dataset, tokenizer)
