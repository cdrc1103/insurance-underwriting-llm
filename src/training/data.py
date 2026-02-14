"""SFT data preparation with label masking for finetuning."""

import logging
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizer

from configs.model import MAX_TOKEN_LENGTH

logger = logging.getLogger(__name__)


def prepare_conversation_for_sft(
    messages: list[dict[str, str]],
    target_response: str,
) -> list[dict[str, str]]:
    """
    Reconstruct the full conversation by appending target response.

    The preprocessing step separates input messages from the target response.
    This function rejoins them into a complete conversation for tokenization.

    Args:
        messages: Input messages (system + user/assistant turns, excluding final response)
        target_response: The final assistant response to train on

    Returns:
        Full conversation messages list including target response

    Raises:
        ValueError: If messages is empty or target_response is empty
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    if not target_response:
        raise ValueError("Target response cannot be empty")

    return messages + [{"role": "assistant", "content": target_response}]


def tokenize_for_sft(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int = MAX_TOKEN_LENGTH,
) -> dict[str, Any]:
    """
    Tokenize a single example for supervised finetuning with label masking.

    Only the target response tokens contribute to the training loss.
    Prompt tokens (system message, conversation history) are masked with -100.

    Args:
        example: Example with 'messages' and 'target_response' fields
        tokenizer: Tokenizer with chat template support
        max_length: Maximum sequence length

    Returns:
        Dictionary with input_ids, attention_mask, and labels (prompt tokens masked)

    Raises:
        ValueError: If required fields are missing
    """
    if "messages" not in example:
        raise ValueError("Example must contain 'messages' field")
    if "target_response" not in example:
        raise ValueError("Example must contain 'target_response' field")

    messages = example["messages"]
    target_response = example["target_response"]

    # Build full conversation (messages + target response)
    full_messages = prepare_conversation_for_sft(messages, target_response)

    # Tokenize full conversation
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )

    # Tokenize prompt only (without target response, with generation prompt)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    prompt_length = len(prompt_tokens["input_ids"])

    # Create labels: -100 for prompt tokens, actual token IDs for response tokens
    labels = [-100] * len(full_tokens["input_ids"])
    for i in range(prompt_length, len(full_tokens["input_ids"])):
        labels[i] = full_tokens["input_ids"][i]

    # Also mask padding tokens in labels
    for i in range(len(labels)):
        if full_tokens["attention_mask"][i] == 0:
            labels[i] = -100

    # Validate that we have trainable tokens
    trainable_count = sum(1 for label in labels if label != -100)
    if trainable_count == 0:
        logger.warning(
            f"Example has no trainable tokens (prompt_length={prompt_length}, "
            f"total_length={len(labels)}). Response may have been fully truncated."
        )

    result = {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }

    # Preserve metadata
    if "original_index" in example:
        result["original_index"] = example["original_index"]
    if "num_turns" in example:
        result["num_turns"] = example["num_turns"]

    return result


def prepare_sft_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = MAX_TOKEN_LENGTH,
) -> Dataset:
    """
    Prepare an entire dataset for supervised finetuning.

    Applies SFT tokenization with label masking to all examples.

    Args:
        dataset: Dataset with 'messages' and 'target_response' columns
        tokenizer: Tokenizer with chat template support
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset ready for training

    Raises:
        ValueError: If dataset is missing required columns
    """
    required_columns = ["messages", "target_response"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset must have '{col}' column. Run preprocessing first.")

    logger.info(f"Preparing SFT dataset: {len(dataset)} examples, max_length={max_length}")

    # Columns to remove after tokenization (keep only model inputs + metadata)
    remove_columns = [
        col for col in dataset.column_names if col not in ["original_index", "num_turns"]
    ]

    def tokenize_fn(example: dict[str, Any]) -> dict[str, Any]:
        return tokenize_for_sft(example, tokenizer, max_length=max_length)

    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=remove_columns,
        desc="Tokenizing for SFT",
    )

    logger.info(f"SFT dataset prepared: {len(tokenized)} examples")
    return tokenized
