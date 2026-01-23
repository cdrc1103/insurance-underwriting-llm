"""Tests for data preprocessing utilities."""

import pytest
from datasets import Dataset

from src.data.preprocessing import (
    clean_text,
    extract_company_profile,
    extract_conversation,
    format_conversation_prompt,
    get_preprocessing_stats,
    has_tool_calls,
    preprocess_dataset,
    preprocess_example,
)


@pytest.fixture
def sample_example():
    """Create a sample example for testing."""
    return {
        "company_name": "Acme Corp",
        "annual_revenue": "$1,000,000",
        "number_of_employees": "50",
        "industry": "Technology",
        "state": "California",
        "messages": [
            {"role": "user", "content": "What insurance do I need?"},
            {"role": "assistant", "content": "Based on your profile, I recommend..."},
        ],
    }


@pytest.fixture
def sample_with_tool_calls():
    """Create a sample example with tool calls."""
    return {
        "company_name": "Test Co",
        "messages": [
            {"role": "user", "content": "Check appetite"},
            {"role": "assistant", "content": "Let me use tool_call to check..."},
        ],
    }


def test_extract_company_profile(sample_example):
    """Test company profile extraction."""
    profile = extract_company_profile(sample_example)

    assert profile["company_name"] == "Acme Corp"
    assert profile["annual_revenue"] == "$1,000,000"
    assert profile["number_of_employees"] == "50"
    assert profile["industry"] == "Technology"
    assert profile["state"] == "California"


def test_extract_company_profile_missing_fields():
    """Test extraction with missing fields."""
    example = {"company_name": "Test"}
    profile = extract_company_profile(example)

    assert profile["company_name"] == "Test"
    assert profile["annual_revenue"] == "Not specified"
    assert profile["number_of_employees"] == "Not specified"


def test_extract_conversation(sample_example):
    """Test conversation extraction."""
    conversation = extract_conversation(sample_example)

    assert len(conversation) == 2
    assert conversation[0]["role"] == "user"
    assert "insurance" in conversation[0]["content"].lower()
    assert conversation[1]["role"] == "assistant"


def test_extract_conversation_invalid():
    """Test extraction with invalid conversation."""
    example = {"no_messages": True}

    with pytest.raises(ValueError, match="No valid conversation"):
        extract_conversation(example)


def test_has_tool_calls():
    """Test tool call detection."""
    example_with_tool = {"text": "Using tool_call to fetch data"}
    example_without_tool = {"text": "Regular conversation"}

    assert has_tool_calls(example_with_tool) is True
    assert has_tool_calls(example_without_tool) is False


def test_clean_text():
    """Test text cleaning."""
    dirty_text = "  Hello    world!!!  Multiple   spaces  "
    clean = clean_text(dirty_text)

    assert clean == "Hello world! Multiple spaces"
    assert clean.strip() == clean
    assert "  " not in clean


def test_clean_text_invalid_type():
    """Test cleaning with invalid input."""
    with pytest.raises(TypeError):
        clean_text(123)


def test_format_conversation_prompt(sample_example):
    """Test conversation prompt formatting."""
    profile = extract_company_profile(sample_example)
    conversation = extract_conversation(sample_example)

    prompt = format_conversation_prompt(profile, conversation)

    assert "Company Profile:" in prompt
    assert "Acme Corp" in prompt
    assert "Conversation:" in prompt
    assert "User:" in prompt
    assert "Assistant:" in prompt


def test_preprocess_example(sample_example):
    """Test preprocessing single example."""
    result = preprocess_example(sample_example)

    assert result is not None
    assert "company_profile" in result
    assert "conversation" in result
    assert "formatted_text" in result
    assert "num_turns" in result
    assert result["num_turns"] == 2


def test_preprocess_example_with_tool_calls(sample_with_tool_calls):
    """Test preprocessing filters tool calls."""
    result = preprocess_example(sample_with_tool_calls, include_tool_calls=False)

    assert result is None


def test_preprocess_example_include_tool_calls(sample_with_tool_calls):
    """Test preprocessing includes tool calls when requested."""
    result = preprocess_example(sample_with_tool_calls, include_tool_calls=True)

    assert result is not None


def test_preprocess_dataset():
    """Test preprocessing entire dataset."""
    data = {
        "company_name": ["Company A", "Company B"],
        "messages": [
            [{"role": "user", "content": "Question 1"}],
            [{"role": "user", "content": "Question 2"}],
        ],
    }

    dataset = Dataset.from_dict(data)
    preprocessed = preprocess_dataset(dataset, verbose=False)

    assert len(preprocessed) > 0
    assert "formatted_text" in preprocessed[0]


def test_preprocess_dataset_empty_after_filtering():
    """Test preprocessing with all examples filtered."""
    data = {
        "company_name": ["Company A"],
        "messages": [[{"role": "user", "content": "tool_call here"}]],
    }

    dataset = Dataset.from_dict(data)

    with pytest.raises(ValueError, match="All examples were filtered"):
        preprocess_dataset(dataset, include_tool_calls=False, verbose=False)


def test_get_preprocessing_stats():
    """Test preprocessing statistics computation."""
    data = [
        {
            "num_turns": 2,
            "formatted_text": "Short text",
        },
        {
            "num_turns": 3,
            "formatted_text": "Longer text here",
        },
    ]

    dataset = Dataset.from_list(data)
    stats = get_preprocessing_stats(dataset)

    assert stats["num_examples"] == 2
    assert stats["turns"]["mean"] == 2.5
    assert stats["turns"]["min"] == 2
    assert stats["turns"]["max"] == 3
    assert "text_length" in stats
