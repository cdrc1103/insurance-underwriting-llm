"""Data preprocessing utilities for insurance underwriting conversations."""

import re
from typing import Any

from datasets import Dataset


def extract_company_profile(example: dict[str, Any]) -> dict[str, Any]:
    """
    Extract company profile information from an example.

    Args:
        example: Raw dataset example

    Returns:
        Dictionary containing company profile fields

    Raises:
        ValueError: If required company fields are missing
    """
    company_profile = {}

    # Extract common company fields (adjust based on actual schema)
    field_mappings = {
        "company_name": ["company_name", "name", "business_name"],
        "annual_revenue": ["annual_revenue", "revenue"],
        "number_of_employees": ["number_of_employees", "employees", "num_employees"],
        "industry": ["industry", "business_type", "business_description"],
        "state": ["state", "location"],
    }

    for target_field, possible_fields in field_mappings.items():
        for field in possible_fields:
            if field in example and example[field] is not None:
                company_profile[target_field] = example[field]
                break
        else:
            # Field not found, set to default
            company_profile[target_field] = "Not specified"

    return company_profile


def extract_conversation(example: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extract multi-turn conversation from an example.

    Args:
        example: Raw dataset example

    Returns:
        List of conversation turns with 'role' and 'content' keys

    Raises:
        ValueError: If conversation structure is invalid
    """
    conversation = []

    # Try different possible conversation structures
    if "messages" in example:
        for msg in example["messages"]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                conversation.append(
                    {
                        "role": msg["role"],
                        "content": clean_text(msg["content"]),
                    }
                )

    elif "conversation" in example:
        for turn in example["conversation"]:
            if isinstance(turn, dict):
                conversation.append(
                    {
                        "role": turn.get("role", "unknown"),
                        "content": clean_text(turn.get("content", "")),
                    }
                )

    if not conversation:
        raise ValueError("No valid conversation found in example")

    return conversation


def has_tool_calls(example: dict[str, Any]) -> bool:
    """
    Check if an example contains tool calls or function calling.

    Args:
        example: Raw dataset example

    Returns:
        True if tool calls are present, False otherwise
    """
    example_str = str(example).lower()

    # Check for common tool call indicators
    tool_indicators = [
        "tool_call",
        "function_call",
        "tool_calls",
        "function_calls",
        "tool_use",
        "function_name",
    ]

    return any(indicator in example_str for indicator in tool_indicators)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text

    Raises:
        TypeError: If text is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove excessive punctuation
    text = re.sub(r"([.!?]){2,}", r"\1", text)

    return text


def format_conversation_prompt(
    company_profile: dict[str, Any],
    conversation: list[dict[str, str]],
) -> str:
    """
    Format company profile and conversation into a standardized prompt.

    Args:
        company_profile: Company information dictionary
        conversation: List of conversation turns

    Returns:
        Formatted prompt string
    """
    # Format company profile
    profile_lines = [
        "Company Profile:",
        f"- Name: {company_profile.get('company_name', 'N/A')}",
        f"- Revenue: {company_profile.get('annual_revenue', 'N/A')}",
        f"- Employees: {company_profile.get('number_of_employees', 'N/A')}",
        f"- Industry: {company_profile.get('industry', 'N/A')}",
        f"- State: {company_profile.get('state', 'N/A')}",
        "",
        "Conversation:",
    ]

    # Format conversation turns
    for turn in conversation:
        role = turn["role"].capitalize()
        content = turn["content"]
        profile_lines.append(f"{role}: {content}")

    return "\n".join(profile_lines)


def preprocess_example(
    example: dict[str, Any], include_tool_calls: bool = True
) -> dict[str, Any] | None:
    """
    Preprocess a single example.

    Args:
        example: Raw dataset example
        include_tool_calls: If True, include examples with tool calls.
            Default is True since we're training Qwen2.5-1.5B-Instruct
            for domain-specific tool use in insurance underwriting.

    Returns:
        Preprocessed example or None if example should be filtered

    Raises:
        ValueError: If example cannot be processed
    """
    # Filter tool calls if requested
    if not include_tool_calls and has_tool_calls(example):
        return None

    try:
        # Extract company profile
        company_profile = extract_company_profile(example)

        # Extract conversation
        conversation = extract_conversation(example)

        # Filter empty conversations
        if not conversation:
            return None

        # Format into standardized structure
        formatted_text = format_conversation_prompt(company_profile, conversation)

        return {
            "company_profile": company_profile,
            "conversation": conversation,
            "formatted_text": formatted_text,
            "num_turns": len(conversation),
            "original_index": example.get("index", -1),
        }

    except (ValueError, KeyError, TypeError) as e:
        # Log warning and skip problematic examples
        print(f"Warning: Failed to preprocess example: {e}")
        return None


def preprocess_dataset(
    dataset: Dataset,
    include_tool_calls: bool = True,
    verbose: bool = True,
) -> Dataset:
    """
    Preprocess entire dataset.

    Args:
        dataset: Raw dataset to preprocess
        include_tool_calls: If True, include examples with tool calls.
            Default is True since we're training Qwen2.5-1.5B-Instruct
            for domain-specific tool use in insurance underwriting.
        verbose: If True, print preprocessing statistics

    Returns:
        Preprocessed dataset

    Raises:
        ValueError: If dataset is empty after preprocessing
    """
    if verbose:
        print(f"Preprocessing {len(dataset)} examples...")

    # Preprocess examples
    preprocessed = []
    filtered_count = 0

    for idx, example in enumerate(dataset):
        # Add index to example
        example["index"] = idx

        # Preprocess
        processed = preprocess_example(example, include_tool_calls=include_tool_calls)

        if processed is not None:
            preprocessed.append(processed)
        else:
            filtered_count += 1

    if not preprocessed:
        raise ValueError("All examples were filtered during preprocessing")

    if verbose:
        print("Preprocessing complete:")
        print(f"  - Kept: {len(preprocessed)} examples")
        print(f"  - Filtered: {filtered_count} examples")
        print(f"  - Retention rate: {len(preprocessed) / len(dataset) * 100:.1f}%")

    # Convert to Dataset
    return Dataset.from_list(preprocessed)


def get_preprocessing_stats(dataset: Dataset) -> dict[str, Any]:
    """
    Compute statistics about preprocessed dataset.

    Args:
        dataset: Preprocessed dataset

    Returns:
        Dictionary containing preprocessing statistics
    """
    num_turns = [ex["num_turns"] for ex in dataset]
    text_lengths = [len(ex["formatted_text"]) for ex in dataset]

    stats = {
        "num_examples": len(dataset),
        "turns": {
            "mean": sum(num_turns) / len(num_turns),
            "min": min(num_turns),
            "max": max(num_turns),
        },
        "text_length": {
            "mean": sum(text_lengths) / len(text_lengths),
            "min": min(text_lengths),
            "max": max(text_lengths),
        },
    }

    return stats
