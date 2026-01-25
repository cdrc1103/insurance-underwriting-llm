"""Tokenization utilities for insurance underwriting conversations."""

from typing import Any

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from configs.model import QWEN_MAX_CONTEXT


def load_tokenizer(
    model_name: str = "Qwen/Qwen3-0.6B",
    add_special_tokens: bool = True,
) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer for a language model.

    Args:
        model_name: Name or path of the model
        add_special_tokens: Whether to add special tokens

    Returns:
        Configured tokenizer

    Raises:
        ValueError: If tokenizer cannot be loaded
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not set (common for some models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add special tokens if requested
        if add_special_tokens:
            # Add tokens for conversation roles if not present
            special_tokens = {"additional_special_tokens": ["[USER]", "[ASSISTANT]", "[COMPANY]"]}
            tokenizer.add_special_tokens(special_tokens)

        return tokenizer

    except Exception as e:
        raise ValueError(f"Failed to load tokenizer '{model_name}': {e}") from e


def format_conversation_for_training(example: dict[str, Any]) -> str:
    """
    Format a conversation example for training.

    Args:
        example: Preprocessed example with conversation

    Returns:
        Formatted text ready for tokenization
    """
    # Start with formatted text from preprocessing
    if "formatted_text" in example:
        return example["formatted_text"]

    # Fallback: build from conversation
    parts = []

    # Add company profile if available
    if "company_profile" in example:
        profile = example["company_profile"]
        parts.append("Company Profile:")
        for key, value in profile.items():
            parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        parts.append("\nConversation:")

    # Add conversation turns
    if "conversation" in example:
        for turn in example["conversation"]:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")

    return "\n".join(parts)


def tokenize_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    truncation: bool = True,
    padding: str = "max_length",
) -> dict[str, Any]:
    """
    Tokenize a single example.

    Args:
        example: Example to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        truncation: Whether to truncate long sequences
        padding: Padding strategy ("max_length", "longest", or False)

    Returns:
        Dictionary with tokenized inputs
    """
    # Format conversation
    text = format_conversation_for_training(example)

    # Tokenize
    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=None,  # Return lists, not tensors
    )

    # Add labels (for causal language modeling, labels = input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Store metadata
    tokenized["original_index"] = example.get("original_index", -1)
    tokenized["num_turns"] = example.get("num_turns", 0)

    return tokenized


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    truncation: bool = True,
    padding: str = "max_length",
    remove_columns: list[str] | None = None,
    verbose: bool = True,
) -> Dataset:
    """
    Tokenize entire dataset.

    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        truncation: Whether to truncate long sequences
        padding: Padding strategy
        remove_columns: Columns to remove after tokenization
        verbose: Whether to print progress

    Returns:
        Tokenized dataset
    """
    if verbose:
        print(f"Tokenizing {len(dataset)} examples...")
        print(f"  Max length: {max_length}")
        print(f"  Truncation: {truncation}")
        print(f"  Padding: {padding}")

    # Tokenize function
    def tokenize_fn(example):
        return tokenize_example(
            example,
            tokenizer,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )

    # Apply tokenization
    if remove_columns is None:
        # Remove all columns except metadata
        remove_columns = [
            col for col in dataset.column_names if col not in ["original_index", "num_turns"]
        ]

    tokenized_dataset = dataset.map(
        tokenize_fn,
        remove_columns=remove_columns,
        desc="Tokenizing" if verbose else None,
    )

    if verbose:
        print(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")

    return tokenized_dataset


def compute_token_statistics(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Any]:
    """
    Compute token count statistics for a dataset.

    Args:
        dataset: Dataset (tokenized or raw)
        tokenizer: Tokenizer used

    Returns:
        Dictionary with token statistics
    """
    # Check if already tokenized
    if "input_ids" in dataset.column_names:
        token_counts = [len(ids) for ids in dataset["input_ids"]]
    else:
        # Tokenize on the fly to get counts
        token_counts = []
        for example in dataset:
            text = format_conversation_for_training(example)
            tokens = tokenizer(text, return_tensors=None)
            token_counts.append(len(tokens["input_ids"]))

    stats = {
        "total_examples": len(dataset),
        "mean": float(np.mean(token_counts)),
        "median": float(np.median(token_counts)),
        "min": int(np.min(token_counts)),
        "max": int(np.max(token_counts)),
        "std": float(np.std(token_counts)),
        "percentiles": {
            "25th": float(np.percentile(token_counts, 25)),
            "50th": float(np.percentile(token_counts, 50)),
            "75th": float(np.percentile(token_counts, 75)),
            "90th": float(np.percentile(token_counts, 90)),
            "95th": float(np.percentile(token_counts, 95)),
            "99th": float(np.percentile(token_counts, 99)),
        },
    }

    return stats


def mark_truncated_examples(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = QWEN_MAX_CONTEXT,
    verbose: bool = True,
) -> tuple[Dataset, dict[str, Any]]:
    """
    Mark examples that will be truncated during tokenization.

    This function adds metadata columns to track which examples exceed the
    maximum token length and by how much. The actual truncation happens
    during tokenization - this provides visibility into what will be affected.

    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer to use for token counting
        max_length: Maximum sequence length (default: Qwen3 32K context)
        verbose: Whether to print truncation statistics

    Returns:
        Tuple of (dataset with truncation metadata, statistics dict)

    Raises:
        ValueError: If dataset is empty
    """
    if len(dataset) == 0:
        raise ValueError("Cannot process empty dataset")

    token_counts = []

    # Compute token counts for each example
    for example in dataset:
        text = format_conversation_for_training(example)
        tokens = tokenizer(text, return_tensors=None)
        token_counts.append(len(tokens["input_ids"]))

    # Determine truncation status for each example
    will_truncate = [count > max_length for count in token_counts]
    tokens_over_limit = [max(0, count - max_length) for count in token_counts]

    # Add metadata columns to dataset
    dataset = dataset.add_column("token_count", token_counts)
    dataset = dataset.add_column("will_truncate", will_truncate)
    dataset = dataset.add_column("tokens_over_limit", tokens_over_limit)

    # Compute statistics
    truncated_count = sum(will_truncate)
    truncated_indices = [i for i, t in enumerate(will_truncate) if t]

    stats = {
        "max_length": max_length,
        "total_examples": len(dataset),
        "truncated_count": truncated_count,
        "truncated_percentage": (truncated_count / len(dataset)) * 100,
        "truncated_indices": truncated_indices,
        "max_tokens_over": max(tokens_over_limit) if tokens_over_limit else 0,
        "avg_tokens_over": (
            sum(t for t in tokens_over_limit if t > 0) / truncated_count
            if truncated_count > 0
            else 0
        ),
    }

    if verbose:
        print(f"Truncation analysis (max_length={max_length}):")
        print(f"  Total examples: {stats['total_examples']}")
        print(
            f"  Will be truncated: {stats['truncated_count']} ({stats['truncated_percentage']:.1f}%)"
        )
        if truncated_count > 0:
            print(f"  Max tokens over limit: {stats['max_tokens_over']}")
            print(f"  Avg tokens over limit: {stats['avg_tokens_over']:.0f}")

    return dataset, stats


def get_recommended_max_length(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    percentile: float = 95.0,
) -> int:
    """
    Get recommended max length based on token distribution.

    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer to use
        percentile: Percentile to use for recommendation (default: 95th)

    Returns:
        Recommended max length

    Raises:
        ValueError: If percentile is not between 0 and 100
    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

    stats = compute_token_statistics(dataset, tokenizer)
    percentile_key = f"{int(percentile)}th"

    if percentile_key in stats["percentiles"]:
        return int(stats["percentiles"][percentile_key])
    else:
        # Fallback to mean + 2*std
        return int(stats["mean"] + 2 * stats["std"])
