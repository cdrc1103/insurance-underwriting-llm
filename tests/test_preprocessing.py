"""Tests for data preprocessing utilities."""

import pytest
from datasets import Dataset

from src.data.preprocessing import (
    clean_text,
    extract_company_profile,
    extract_conversation,
    format_messages_for_training,
    format_system_prompt,
    get_preprocessing_stats,
    has_tool_calls,
    preprocess_dataset,
    preprocess_example,
)


@pytest.fixture
def sample_example():
    """Create a sample example matching the Multi-Turn-Insurance-Underwriting schema."""
    return {
        "primary id": 0,
        "company task id": 1097,
        "assistant model name": "o3",
        "task": "Product Recommendations",
        "company name": "Acme Corp",
        "annual revenue": 1000000,
        "number of employees": 50,
        "total payroll": 2500000,
        "number of vehicles": 5,
        "building construction": "Non-combustible",
        "state": "California",
        "company description": "A technology company",
        "lob": "general liability",
        "reference answer": "Recommended products: GL, Property",
        "correct": True,
        "trace": [
            {
                "role": "user",
                "content": "What insurance do I need?",
                "type": "underwriter",
                "tool_calls": "",
            },
            {
                "role": "assistant",
                "content": "Based on your profile, I recommend general liability.",
                "type": "user-facing assistant",
                "tool_calls": "",
            },
        ],
    }


@pytest.fixture
def sample_with_tool_calls():
    """Create a sample example with tool calls."""
    return {
        "company name": "Test Co",
        "annual revenue": 500000,
        "number of employees": 10,
        "total payroll": 400000,
        "number of vehicles": 2,
        "building construction": "Wood Frame",
        "state": "Texas",
        "company description": "A small business",
        "lob": "property",
        "task": "Appetite Check",
        "reference answer": "In appetite",
        "correct": True,
        "trace": [
            {
                "role": "user",
                "content": "Check appetite for this company",
                "type": "underwriter",
                "tool_calls": "",
            },
            {
                "role": "assistant",
                "content": "",
                "type": "internal assistant",
                "tool_calls": "[{'name': 'get_underwriting_guidelines', 'args': {}}]",
            },
            {
                "role": "assistant",
                "content": "Underwriting guidelines content here...",
                "type": "tool",
                "tool_calls": "",
            },
            {
                "role": "assistant",
                "content": "Based on the guidelines, this company is in appetite.",
                "type": "user-facing assistant",
                "tool_calls": "",
            },
        ],
    }


@pytest.fixture
def sample_with_internal_reasoning():
    """Create a sample example with internal assistant reasoning (non-empty content)."""
    return {
        "company name": "Reasoning Corp",
        "annual revenue": 750000,
        "number of employees": 25,
        "total payroll": 600000,
        "number of vehicles": 3,
        "building construction": "Masonry",
        "state": "New York",
        "company description": "A consulting firm",
        "lob": "professional liability",
        "task": "Coverage Assessment",
        "reference answer": "E&O coverage recommended",
        "correct": True,
        "trace": [
            {
                "role": "user",
                "content": "What coverage does this company need?",
                "type": "underwriter",
                "tool_calls": "",
            },
            {
                "role": "assistant",
                "content": "Let me analyze the company profile and check guidelines.",
                "type": "internal assistant",
                "tool_calls": "",
            },
            {
                "role": "assistant",
                "content": "Based on the analysis, E&O coverage is recommended.",
                "type": "user-facing assistant",
                "tool_calls": "",
            },
        ],
    }


def test_extract_company_profile(sample_example):
    """Test company profile extraction."""
    profile = extract_company_profile(sample_example)

    assert profile["company_name"] == "Acme Corp"
    assert profile["annual_revenue"] == 1000000
    assert profile["number_of_employees"] == 50
    assert profile["total_payroll"] == 2500000
    assert profile["number_of_vehicles"] == 5
    assert profile["building_construction"] == "Non-combustible"
    assert profile["state"] == "California"
    assert profile["company_description"] == "A technology company"
    assert profile["lob"] == "general liability"


def test_extract_company_profile_missing_fields():
    """Test extraction with missing fields."""
    example = {"company name": "Test"}
    profile = extract_company_profile(example)

    assert profile["company_name"] == "Test"
    assert profile["annual_revenue"] is None
    assert profile["number_of_employees"] is None
    assert profile["state"] is None


def test_extract_conversation(sample_example):
    """Test conversation extraction from trace."""
    conversation = extract_conversation(sample_example)

    assert len(conversation) == 2
    assert conversation[0]["role"] == "user"
    assert "insurance" in conversation[0]["content"].lower()
    assert conversation[0]["type"] == "underwriter"
    assert conversation[1]["role"] == "assistant"
    assert conversation[1]["type"] == "user-facing assistant"


def test_extract_conversation_with_tool_calls(sample_with_tool_calls):
    """Test conversation extraction with tool calls and tool responses."""
    conversation = extract_conversation(sample_with_tool_calls)

    # Should skip empty content messages
    assert len(conversation) == 3
    assert conversation[0]["role"] == "user"
    assert conversation[1]["role"] == "tool"
    assert conversation[2]["role"] == "assistant"


def test_extract_conversation_missing_trace():
    """Test extraction raises error when trace field is missing."""
    example = {"company name": "Test Corp"}

    with pytest.raises(ValueError, match="missing required 'trace' field"):
        extract_conversation(example)


def test_extract_conversation_empty_trace():
    """Test extraction raises error when trace has no valid messages."""
    example = {"trace": [{"content": "", "type": "underwriter"}]}

    with pytest.raises(ValueError, match="No valid conversation"):
        extract_conversation(example)


def test_has_tool_calls():
    """Test tool call detection."""
    example_with_tool = {
        "trace": [
            {"role": "user", "content": "Check guidelines", "tool_calls": ""},
            {"role": "assistant", "content": "", "tool_calls": "get_underwriting_guidelines()"},
        ]
    }
    example_without_tool = {
        "trace": [
            {"role": "user", "content": "Hello", "tool_calls": ""},
            {"role": "assistant", "content": "Hi there", "tool_calls": ""},
        ]
    }
    example_no_trace = {"text": "No trace field"}

    assert has_tool_calls(example_with_tool) is True
    assert has_tool_calls(example_without_tool) is False
    assert has_tool_calls(example_no_trace) is False


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


def test_format_system_prompt(sample_example):
    """Test system prompt formatting with company profile."""
    profile = extract_company_profile(sample_example)
    system_prompt = format_system_prompt(profile)

    assert "insurance underwriting co-pilot" in system_prompt
    # Check all tools are listed
    assert "get_underwriting_guidelines" in system_prompt
    assert "read_query" in system_prompt
    assert "list_tables" in system_prompt
    assert "get_table_schema" in system_prompt
    assert "## Company Profile" in system_prompt
    assert "Acme Corp" in system_prompt
    assert "$1,000,000" in system_prompt  # Revenue formatted
    assert "California" in system_prompt


def test_format_messages_for_training(sample_example):
    """Test messages formatting for Qwen training."""
    profile = extract_company_profile(sample_example)
    conversation = extract_conversation(sample_example)

    messages = format_messages_for_training(profile, conversation)

    # Should have system + 2 conversation turns
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"

    # Check system message contains company info
    assert "Acme Corp" in messages[0]["content"]

    # Check conversation content
    assert "insurance" in messages[1]["content"].lower()
    assert "liability" in messages[2]["content"].lower()


def test_format_messages_with_tool_response(sample_with_tool_calls):
    """Test messages formatting includes tool responses."""
    profile = extract_company_profile(sample_with_tool_calls)
    conversation = extract_conversation(sample_with_tool_calls)

    messages = format_messages_for_training(profile, conversation)

    # System + user + tool + assistant
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "tool"
    assert messages[3]["role"] == "assistant"


def test_format_messages_with_thinking_mode(sample_with_internal_reasoning):
    """Test that internal assistant content is wrapped in <think> tags (Qwen3 format)."""
    profile = extract_company_profile(sample_with_internal_reasoning)
    conversation = extract_conversation(sample_with_internal_reasoning)

    messages = format_messages_for_training(profile, conversation)

    # System + user + combined assistant (internal + user-facing)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"

    # Check that internal reasoning is wrapped in <think> tags
    assistant_content = messages[2]["content"]
    assert "<think>" in assistant_content
    assert "</think>" in assistant_content
    assert "analyze the company profile" in assistant_content
    assert "E&O coverage is recommended" in assistant_content

    # Verify structure: <think>internal</think>user-facing
    assert assistant_content.startswith("<think>")
    assert "</think>" in assistant_content
    think_end = assistant_content.index("</think>")
    user_facing_part = assistant_content[think_end + len("</think>") :]
    assert "E&O coverage is recommended" in user_facing_part


def test_preprocess_example(sample_example):
    """Test preprocessing single example produces messages format."""
    result = preprocess_example(sample_example)

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 3  # system + 2 turns
    assert result["messages"][0]["role"] == "system"
    assert result["num_turns"] == 2
    assert result["num_user_turns"] == 1
    assert result["num_assistant_turns"] == 1
    assert result["task"] == "Product Recommendations"
    assert result["reference_answer"] == "Recommended products: GL, Property"
    assert result["correct"] is True


def test_preprocess_example_with_tool_calls(sample_with_tool_calls):
    """Test preprocessing includes examples with tool calls."""
    result = preprocess_example(sample_with_tool_calls)

    assert result is not None
    assert "messages" in result
    assert result["num_tool_turns"] == 1

    # Verify tool message is present
    tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1


def test_preprocess_example_empty_trace():
    """Test preprocessing with empty trace returns None."""
    example = {
        "company name": "Empty Corp",
        "trace": [],
    }
    result = preprocess_example(example)
    assert result is None


def test_preprocess_dataset():
    """Test preprocessing entire dataset."""
    data = {
        "company name": ["Company A", "Company B"],
        "annual revenue": [100000, 200000],
        "number of employees": [10, 20],
        "total payroll": [50000, 100000],
        "number of vehicles": [1, 2],
        "building construction": ["Wood", "Steel"],
        "state": ["CA", "TX"],
        "company description": ["Desc A", "Desc B"],
        "lob": ["property", "auto"],
        "task": ["Appetite Check", "Policy Limits"],
        "reference answer": ["In appetite", "Limit: $1M"],
        "correct": [True, False],
        "trace": [
            [{"role": "user", "content": "Question 1", "type": "underwriter", "tool_calls": ""}],
            [{"role": "user", "content": "Question 2", "type": "underwriter", "tool_calls": ""}],
        ],
    }

    dataset = Dataset.from_dict(data)
    preprocessed = preprocess_dataset(dataset, verbose=False)

    assert len(preprocessed) == 2
    assert "messages" in preprocessed[0]
    assert "task" in preprocessed[0]


def test_preprocess_dataset_with_tool_calls():
    """Test preprocessing dataset with tool call examples."""
    data = {
        "company name": ["Company A"],
        "annual revenue": [100000],
        "number of employees": [10],
        "total payroll": [50000],
        "number of vehicles": [1],
        "building construction": ["Wood"],
        "state": ["CA"],
        "company description": ["Description"],
        "lob": ["property"],
        "task": ["Appetite Check"],
        "reference answer": ["Answer"],
        "correct": [True],
        "trace": [
            [
                {
                    "role": "user",
                    "content": "Check appetite",
                    "type": "underwriter",
                    "tool_calls": "",
                },
                {
                    "role": "assistant",
                    "content": "Guidelines here",
                    "type": "tool",
                    "tool_calls": "",
                },
            ]
        ],
    }

    dataset = Dataset.from_dict(data)
    preprocessed = preprocess_dataset(dataset, verbose=False)

    assert len(preprocessed) == 1
    assert "messages" in preprocessed[0]


def test_get_preprocessing_stats():
    """Test preprocessing statistics computation."""
    data = [
        {
            "num_turns": 2,
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Short"},
                {"role": "assistant", "content": "Reply"},
            ],
        },
        {
            "num_turns": 3,
            "messages": [
                {"role": "system", "content": "System prompt here"},
                {"role": "user", "content": "Longer question"},
                {"role": "assistant", "content": "Longer reply here"},
                {"role": "user", "content": "Follow up"},
            ],
        },
    ]

    dataset = Dataset.from_list(data)
    stats = get_preprocessing_stats(dataset)

    assert stats["num_examples"] == 2
    assert stats["turns"]["mean"] == 2.5
    assert stats["turns"]["min"] == 2
    assert stats["turns"]["max"] == 3
    assert "content_length" in stats
