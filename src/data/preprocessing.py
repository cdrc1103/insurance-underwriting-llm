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
    """
    # Dataset uses consistent space-separated field names
    return {
        "company_name": example.get("company name"),
        "annual_revenue": example.get("annual revenue"),
        "number_of_employees": example.get("number of employees"),
        "total_payroll": example.get("total payroll"),
        "number_of_vehicles": example.get("number of vehicles"),
        "building_construction": example.get("building construction"),
        "state": example.get("state"),
        "company_description": example.get("company description"),
        "lob": example.get("lob"),
    }


def extract_conversation(example: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extract multi-turn conversation from an example.

    The dataset uses a 'trace' field with messages that have:
    - role: 'user' or 'assistant'
    - content: message text
    - type: 'underwriter', 'user-facing assistant', 'internal assistant', 'tool'
    - tool_calls: string with tool call info (may be empty)

    Args:
        example: Raw dataset example

    Returns:
        List of conversation turns with 'role', 'content', and 'type' keys

    Raises:
        ValueError: If conversation structure is invalid
    """
    if "trace" not in example:
        raise ValueError("Example missing required 'trace' field")

    conversation = []

    for msg in example["trace"]:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", "")
        msg_type = msg.get("type", "")
        role = msg.get("role", "")

        # Skip empty content messages (often tool call requests)
        if not content or not content.strip():
            continue

        # Map message types to standardized roles
        # 'underwriter' -> user (the human asking questions)
        # 'user-facing assistant' -> assistant (the model's response to user)
        # 'internal assistant' -> assistant (internal reasoning, may have tool calls)
        # 'tool' -> tool (tool response)
        if msg_type == "underwriter":
            standardized_role = "user"
        elif msg_type in ("user-facing assistant", "internal assistant"):
            standardized_role = "assistant"
        elif msg_type == "tool":
            standardized_role = "tool"
        else:
            # Fall back to the role field
            standardized_role = role if role else "unknown"

        conversation.append(
            {
                "role": standardized_role,
                "content": clean_text(content),
                "type": msg_type,
                "has_tool_calls": bool(msg.get("tool_calls")),
            }
        )

    if not conversation:
        raise ValueError("No valid conversation found in trace")

    return conversation


def has_tool_calls(example: dict[str, Any]) -> bool:
    """
    Check if an example contains tool calls.

    Args:
        example: Raw dataset example

    Returns:
        True if tool calls are present, False otherwise
    """
    # Check trace messages for non-empty tool_calls field
    if "trace" not in example:
        return False

    for msg in example["trace"]:
        if isinstance(msg, dict) and msg.get("tool_calls"):
            return True

    return False


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


def format_system_prompt(company_profile: dict[str, Any]) -> str:
    """
    Format company profile into a system prompt for Qwen.

    Args:
        company_profile: Company information dictionary

    Returns:
        System prompt string with company context
    """

    def format_value(value: Any, is_currency: bool = False) -> str:
        """Format a value for display, handling None and numeric types."""
        if value is None:
            return "N/A"
        if isinstance(value, int):
            if is_currency:
                return f"${value:,}"
            elif value > 1000:
                return f"{value:,}"
        return str(value)

    profile_lines = [
        "You are an insurance underwriting co-pilot. You assist underwriters by "
        "reasoning about company eligibility, recommending insurance products, "
        "suggesting policy limits and deductibles, and making coverage decisions.",
        "",
        "## Tools",
        "",
        "You have access to the following tools:",
        "",
        "### Database Tools",
        "- list_tables: List available tables in the underwriting database",
        "- get_table_schema: Get the schema/structure of a specific table",
        "- get_table_descriptions: Get descriptions of database tables",
        "- get_table_data_dictionary: Get the data dictionary for table columns",
        "- read_query: Execute SQL queries to retrieve appetite matrix data, "
        "policy information, NAICS classifications, and historical records",
        "",
        "### Guidelines Tool",
        "- get_underwriting_guidelines: Retrieve underwriting guidelines, "
        "eligibility criteria, and coverage requirements",
        "",
        "When answering questions, use these tools to retrieve accurate information "
        "before providing recommendations.",
        "",
        "## Company Profile",
        "",
        f"- Name: {format_value(company_profile.get('company_name'))}",
        f"- Annual Revenue: {format_value(company_profile.get('annual_revenue'), is_currency=True)}",
        f"- Employees: {format_value(company_profile.get('number_of_employees'))}",
        f"- Total Payroll: {format_value(company_profile.get('total_payroll'), is_currency=True)}",
        f"- Vehicles: {format_value(company_profile.get('number_of_vehicles'))}",
        f"- Building Construction: {format_value(company_profile.get('building_construction'))}",
        f"- State: {format_value(company_profile.get('state'))}",
        f"- Line of Business: {format_value(company_profile.get('lob'))}",
    ]

    if company_profile.get("company_description"):
        profile_lines.append(f"- Description: {company_profile['company_description']}")

    return "\n".join(profile_lines)


def format_messages_for_training(
    company_profile: dict[str, Any],
    conversation: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Format company profile and conversation into Qwen3 ChatML messages format.

    This produces the format expected by Qwen3 for fine-tuning with thinking mode:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>internal reasoning</think>user-facing response"},
        {"role": "tool", "content": "..."},
        ...
    ]

    Internal assistant messages are wrapped in <think>...</think> tags and combined
    with subsequent user-facing assistant messages into a single assistant turn.
    This leverages Qwen3's native thinking mode support.

    Args:
        company_profile: Company information dictionary
        conversation: List of conversation turns from extract_conversation

    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    messages = []

    # Add system message with company profile
    system_prompt = format_system_prompt(company_profile)
    messages.append({"role": "system", "content": system_prompt})

    # Process conversation turns, combining internal + user-facing assistant turns
    i = 0
    while i < len(conversation):
        turn = conversation[i]
        role = turn["role"]
        content = turn["content"]

        if role == "assistant":
            # Collect consecutive assistant turns and combine them
            combined_content = []

            while i < len(conversation) and conversation[i]["role"] == "assistant":
                current_turn = conversation[i]
                current_type = current_turn.get("type", "")
                current_content = current_turn["content"]

                if current_type == "internal assistant":
                    # Wrap internal reasoning in <think> tags
                    combined_content.append(f"<think>{current_content}</think>")
                else:
                    # User-facing assistant content goes directly
                    combined_content.append(current_content)

                i += 1

            # Join all parts into single assistant message
            messages.append({"role": "assistant", "content": "".join(combined_content)})
        else:
            # User or tool messages pass through unchanged
            messages.append({"role": role, "content": content})
            i += 1

    return messages


def preprocess_example(example: dict[str, Any]) -> dict[str, Any] | None:
    """
    Preprocess a single example from the Multi-Turn-Insurance-Underwriting dataset.

    Produces output in Qwen3 ChatML messages format with thinking mode support.

    Args:
        example: Raw dataset example containing trace, company info, and metadata

    Returns:
        Preprocessed example with the following fields, or None if invalid:
            - messages: List of ChatML messages for training
            - num_turns: Total conversation turns (excluding system message)
            - num_user_turns: Number of user messages
            - num_assistant_turns: Number of assistant messages
            - num_tool_turns: Number of tool response messages
            - original_index: Index from original dataset
            - task: Task type (e.g., "appetite check", "product recommendation")
            - reference_answer: Ground truth answer for evaluation
            - correct: Whether the original response matched reference (LLM-as-judge)
            - assistant_model: Model that generated the original response
    """
    try:
        # Extract company profile
        company_profile = extract_company_profile(example)

        # Extract conversation from trace
        conversation = extract_conversation(example)

        # Filter empty conversations
        if not conversation:
            return None

        # Format into Qwen3 messages format with thinking mode for training
        messages = format_messages_for_training(company_profile, conversation)

        # Count different message types (excluding system message)
        user_turns = sum(1 for msg in messages if msg["role"] == "user")
        assistant_turns = sum(1 for msg in messages if msg["role"] == "assistant")
        tool_turns = sum(1 for msg in messages if msg["role"] == "tool")

        return {
            # Primary training field - Qwen3 ChatML format with thinking mode
            "messages": messages,
            # Metadata for analysis and filtering
            "num_turns": len(messages) - 1,  # Exclude system message
            "num_user_turns": user_turns,
            "num_assistant_turns": assistant_turns,
            "num_tool_turns": tool_turns,
            "original_index": example.get("index", example.get("primary id", -1)),
            "task": example.get("task", ""),
            "reference_answer": example.get("reference answer", ""),
            "correct": example.get("correct", None),
            "assistant_model": example.get("assistant model name", ""),
        }

    except (ValueError, KeyError, TypeError) as e:
        # Log warning and skip problematic examples
        print(f"Warning: Failed to preprocess example: {e}")
        return None


def preprocess_dataset(
    dataset: Dataset,
    verbose: bool = True,
) -> Dataset:
    """
    Preprocess entire dataset.

    Args:
        dataset: Raw dataset to preprocess
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
    failed_count = 0

    for idx, example in enumerate(dataset):
        # Add index to example
        example["index"] = idx

        # Preprocess
        processed = preprocess_example(example)

        if processed is not None:
            preprocessed.append(processed)
        else:
            failed_count += 1

    if not preprocessed:
        raise ValueError("All examples were filtered during preprocessing")

    if verbose:
        print("Preprocessing complete:")
        print(f"  - Processed: {len(preprocessed)} examples")
        print(f"  - Failed: {failed_count} examples")
        print(f"  - Success rate: {len(preprocessed) / len(dataset) * 100:.1f}%")

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

    # Calculate total content length from messages
    def get_messages_length(ex: dict) -> int:
        return sum(len(msg["content"]) for msg in ex["messages"])

    content_lengths = [get_messages_length(ex) for ex in dataset]

    stats = {
        "num_examples": len(dataset),
        "turns": {
            "mean": sum(num_turns) / len(num_turns),
            "min": min(num_turns),
            "max": max(num_turns),
        },
        "content_length": {
            "mean": sum(content_lengths) / len(content_lengths),
            "min": min(content_lengths),
            "max": max(content_lengths),
        },
    }

    return stats
