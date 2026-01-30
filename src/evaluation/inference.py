"""Inference pipeline for generating model responses.

This module provides a reusable inference pipeline for generating responses
from language models on insurance underwriting conversations. It supports:

- Configurable generation parameters (temperature, top_p, top_k, etc.)
- Custom stopping criteria
- Batched inference for efficiency
- Generation parameter logging for reproducibility
- Response extraction and metadata tracking
"""

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "EvaluationResult",
    "generate_response",
    "generate_response_with_metadata",
    "evaluate_dataset",
    "batch_generate_responses",
    "evaluate_dataset_batched",
    "save_evaluation_results",
    "format_prompt_for_inference",
    "extract_response_content",
]

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        stop_strings: List of strings that stop generation when encountered
    """

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_strings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0], got {self.top_p}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stop_strings": self.stop_strings,
        }


@dataclass
class GenerationResult:
    """Result from a single generation.

    Attributes:
        response: Generated response text
        generation_time_ms: Time taken to generate in milliseconds
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens generated
        config: Generation config used
    """

    response: str
    generation_time_ms: float
    input_tokens: int
    output_tokens: int
    config: dict[str, Any]


class StopStringsCriteria(StoppingCriteria):
    """Stopping criteria that stops generation when any stop string is encountered."""

    def __init__(self, tokenizer: PreTrainedTokenizer, stop_strings: list[str], input_length: int):
        """
        Initialize stop strings criteria.

        Args:
            tokenizer: Tokenizer for decoding generated tokens
            stop_strings: List of strings that trigger stopping
            input_length: Length of input tokens to skip when checking
        """
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_length = input_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> bool:
        """Check if any stop string is in generated text."""
        if not self.stop_strings:
            return False

        # Decode only the generated portion
        generated_ids = input_ids[0, self.input_length :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True

        return False


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    config: GenerationConfig | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stop_strings: list[str] | None = None,
) -> str:
    """
    Generate response from model given conversation messages.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        messages: List of message dicts with 'role' and 'content' keys
        config: GenerationConfig object (overrides individual parameters if provided)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        stop_strings: List of strings that stop generation when encountered

    Returns:
        Generated response text

    Raises:
        ValueError: If generation fails
    """
    # Use config if provided, otherwise build from individual params
    if config is not None:
        max_new_tokens = config.max_new_tokens
        temperature = config.temperature
        top_p = config.top_p
        top_k = config.top_k
        repetition_penalty = config.repetition_penalty
        stop_strings = config.stop_strings
    elif stop_strings is None:
        stop_strings = []

    try:
        # Format messages using chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        # Build stopping criteria
        stopping_criteria = None
        if stop_strings:
            stopping_criteria = StoppingCriteriaList(
                [StopStringsCriteria(tokenizer, stop_strings, input_length)]
            )

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k if temperature > 0 else None,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

        # Decode response (skip the input prompt)
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        # Trim response at stop strings if present
        response = _trim_at_stop_strings(response, stop_strings)

        return response.strip()

    except Exception as e:
        raise ValueError(f"Generation failed: {e}") from e


def _trim_at_stop_strings(text: str, stop_strings: list[str]) -> str:
    """
    Trim text at the first occurrence of any stop string.

    Args:
        text: Text to trim
        stop_strings: List of strings to trim at

    Returns:
        Trimmed text
    """
    if not stop_strings:
        return text

    min_index = len(text)
    for stop_string in stop_strings:
        index = text.find(stop_string)
        if index != -1 and index < min_index:
            min_index = index

    return text[:min_index]


def generate_response_with_metadata(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    config: GenerationConfig | None = None,
) -> GenerationResult:
    """
    Generate response with detailed metadata for logging and analysis.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        messages: List of message dicts with 'role' and 'content' keys
        config: GenerationConfig object

    Returns:
        GenerationResult with response and metadata

    Raises:
        ValueError: If generation fails
    """
    if config is None:
        config = GenerationConfig()

    # Format messages to get input token count
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_tokens = len(tokenizer.encode(prompt))

    # Time the generation
    start_time = time.perf_counter()
    response = generate_response(model, tokenizer, messages, config=config)
    end_time = time.perf_counter()

    generation_time_ms = (end_time - start_time) * 1000
    output_tokens = len(tokenizer.encode(response))

    return GenerationResult(
        response=response,
        generation_time_ms=round(generation_time_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        config=config.to_dict(),
    )


@dataclass
class EvaluationResult:
    """Result from dataset evaluation.

    Attributes:
        results: List of individual generation results
        config: Generation config used
        total_time_ms: Total evaluation time in milliseconds
        successful_count: Number of successful generations
        failed_count: Number of failed generations
    """

    results: list[dict[str, Any]]
    config: dict[str, Any]
    total_time_ms: float
    successful_count: int
    failed_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": self.results,
            "config": self.config,
            "total_time_ms": self.total_time_ms,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
        }


def evaluate_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Sequence[dict[str, Any]],
    config: GenerationConfig | None = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Generate responses for all examples in dataset.

    Processes examples sequentially with detailed logging and metadata tracking.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        dataset: Dataset with 'messages' field
        config: Generation configuration (uses defaults if not provided)
        verbose: Whether to print progress

    Returns:
        EvaluationResult with all responses and metadata

    Raises:
        ValueError: If dataset is empty or missing 'messages' field
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if config is None:
        config = GenerationConfig()

    if verbose:
        logger.info(f"Starting evaluation on {len(dataset)} examples")
        logger.info(f"Generation config: {config.to_dict()}")

    results = []
    successful_count = 0
    failed_count = 0

    start_time = time.perf_counter()

    for i, example in enumerate(dataset):
        if verbose and i % 10 == 0:
            print(f"Processing example {i}/{len(dataset)}...")

        if "messages" not in example:
            raise ValueError(f"Example {i} missing required 'messages' field")

        try:
            gen_result = generate_response_with_metadata(
                model, tokenizer, example["messages"], config=config
            )

            results.append(
                {
                    "original_index": example.get("original_index", i),
                    "task": example.get("task", "unknown"),
                    "messages": example["messages"],
                    "reference_answer": example.get("reference_answer", ""),
                    "generated_response": gen_result.response,
                    "generation_time_ms": gen_result.generation_time_ms,
                    "input_tokens": gen_result.input_tokens,
                    "output_tokens": gen_result.output_tokens,
                }
            )
            successful_count += 1

        except Exception as e:
            logger.warning(f"Failed to generate response for example {i}: {e}")
            failed_count += 1
            results.append(
                {
                    "original_index": example.get("original_index", i),
                    "task": example.get("task", "unknown"),
                    "messages": example["messages"],
                    "reference_answer": example.get("reference_answer", ""),
                    "generated_response": None,
                    "error": str(e),
                    "generation_time_ms": None,
                    "input_tokens": None,
                    "output_tokens": None,
                }
            )

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    if verbose:
        print(f"Completed evaluation: {successful_count} successful, {failed_count} failed")
        print(f"Total time: {total_time_ms:.2f}ms")

    return EvaluationResult(
        results=results,
        config=config.to_dict(),
        total_time_ms=round(total_time_ms, 2),
        successful_count=successful_count,
        failed_count=failed_count,
    )


def batch_generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages_batch: list[list[dict[str, str]]],
    config: GenerationConfig | None = None,
) -> list[GenerationResult]:
    """
    Generate responses for a batch of conversations.

    Uses batched tokenization and generation for improved efficiency.
    Note: Due to variable-length inputs, this pads sequences which may
    affect memory usage for very heterogeneous batch sizes.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        messages_batch: List of conversation message lists
        config: Generation configuration

    Returns:
        List of GenerationResult objects

    Raises:
        ValueError: If batch is empty or generation fails
    """
    if not messages_batch:
        raise ValueError("Batch is empty")

    if config is None:
        config = GenerationConfig()

    # Format all prompts
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    # Tokenize batch with padding
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Track input lengths for each sequence (before padding)
    input_lengths = [len(tokenizer.encode(p)) for p in prompts]

    # Build stopping criteria (applies to all sequences)
    stopping_criteria = None
    if config.stop_strings:
        # For batched generation, we use the first sequence's length as reference
        stopping_criteria = StoppingCriteriaList(
            [StopStringsCriteria(tokenizer, config.stop_strings, inputs["input_ids"].shape[1])]
        )

    start_time = time.perf_counter()

    # Generate batch
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature if config.temperature > 0 else None,
        top_p=config.top_p if config.temperature > 0 else None,
        top_k=config.top_k if config.temperature > 0 else None,
        repetition_penalty=config.repetition_penalty,
        do_sample=config.temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria,
    )

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    per_example_time_ms = total_time_ms / len(messages_batch)

    # Decode each response
    results = []
    for output_ids, input_len in zip(outputs, input_lengths, strict=True):
        # Find actual start of generated content (skip padding and input)
        # For padded batches, we need to skip from the padded input length
        padded_input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(output_ids[padded_input_len:], skip_special_tokens=True)

        # Trim at stop strings
        response = _trim_at_stop_strings(response, config.stop_strings)
        response = response.strip()

        output_tokens = len(tokenizer.encode(response)) if response else 0

        results.append(
            GenerationResult(
                response=response,
                generation_time_ms=round(per_example_time_ms, 2),
                input_tokens=input_len,
                output_tokens=output_tokens,
                config=config.to_dict(),
            )
        )

    return results


def evaluate_dataset_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Sequence[dict[str, Any]],
    config: GenerationConfig | None = None,
    batch_size: int = 4,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Generate responses for dataset using batched inference for efficiency.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        dataset: Dataset with 'messages' field
        config: Generation configuration
        batch_size: Number of examples to process in each batch
        verbose: Whether to print progress

    Returns:
        EvaluationResult with all responses and metadata

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if config is None:
        config = GenerationConfig()

    if verbose:
        logger.info(
            f"Starting batched evaluation on {len(dataset)} examples (batch_size={batch_size})"
        )
        logger.info(f"Generation config: {config.to_dict()}")

    results = []
    successful_count = 0
    failed_count = 0

    start_time = time.perf_counter()

    # Process in batches
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_examples = [dataset[i] for i in range(batch_start, batch_end)]

        if verbose and batch_start % (batch_size * 5) == 0:
            print(
                f"Processing batch {batch_start // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}..."
            )

        # Validate batch
        for i, example in enumerate(batch_examples):
            if "messages" not in example:
                raise ValueError(f"Example {batch_start + i} missing required 'messages' field")

        try:
            messages_batch = [ex["messages"] for ex in batch_examples]
            gen_results = batch_generate_responses(model, tokenizer, messages_batch, config=config)

            for i, (example, gen_result) in enumerate(
                zip(batch_examples, gen_results, strict=True)
            ):
                results.append(
                    {
                        "original_index": example.get("original_index", batch_start + i),
                        "task": example.get("task", "unknown"),
                        "messages": example["messages"],
                        "reference_answer": example.get("reference_answer", ""),
                        "generated_response": gen_result.response,
                        "generation_time_ms": gen_result.generation_time_ms,
                        "input_tokens": gen_result.input_tokens,
                        "output_tokens": gen_result.output_tokens,
                    }
                )
                successful_count += 1

        except Exception as e:
            logger.warning(f"Batch generation failed, falling back to sequential: {e}")
            # Fall back to sequential for this batch
            for i, example in enumerate(batch_examples):
                try:
                    gen_result = generate_response_with_metadata(
                        model, tokenizer, example["messages"], config=config
                    )
                    results.append(
                        {
                            "original_index": example.get("original_index", batch_start + i),
                            "task": example.get("task", "unknown"),
                            "messages": example["messages"],
                            "reference_answer": example.get("reference_answer", ""),
                            "generated_response": gen_result.response,
                            "generation_time_ms": gen_result.generation_time_ms,
                            "input_tokens": gen_result.input_tokens,
                            "output_tokens": gen_result.output_tokens,
                        }
                    )
                    successful_count += 1
                except Exception as inner_e:
                    logger.warning(f"Failed to generate for example {batch_start + i}: {inner_e}")
                    failed_count += 1
                    results.append(
                        {
                            "original_index": example.get("original_index", batch_start + i),
                            "task": example.get("task", "unknown"),
                            "messages": example["messages"],
                            "reference_answer": example.get("reference_answer", ""),
                            "generated_response": None,
                            "error": str(inner_e),
                            "generation_time_ms": None,
                            "input_tokens": None,
                            "output_tokens": None,
                        }
                    )

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    if verbose:
        print(f"Completed batched evaluation: {successful_count} successful, {failed_count} failed")
        print(f"Total time: {total_time_ms:.2f}ms")

    return EvaluationResult(
        results=results,
        config=config.to_dict(),
        total_time_ms=round(total_time_ms, 2),
        successful_count=successful_count,
        failed_count=failed_count,
    )


def save_evaluation_results(
    evaluation_result: EvaluationResult,
    output_path: Path | str,
    include_messages: bool = True,
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        evaluation_result: EvaluationResult to save
        output_path: Path to save the results
        include_messages: Whether to include full messages in output (can be large)

    Raises:
        IOError: If file cannot be written
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = evaluation_result.to_dict()

    if not include_messages:
        # Remove messages to reduce file size
        for result in data["results"]:
            result.pop("messages", None)

    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise OSError(f"Failed to write evaluation results to {output_path}: {e}") from e

    logger.info(f"Saved evaluation results to {output_path}")


def format_prompt_for_inference(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
) -> str:
    """
    Format conversation messages into a prompt string for inference.

    Uses the tokenizer's chat template to format messages correctly
    for the model, including the generation prompt.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: Model tokenizer with chat template

    Returns:
        Formatted prompt string ready for tokenization

    Raises:
        ValueError: If messages format is invalid
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError("Each message must have 'role' and 'content' keys")

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_response_content(
    response: str,
    remove_thinking: bool = True,
) -> str:
    """
    Extract the user-facing content from a generated response.

    For models using thinking mode (like Qwen3), this removes the
    internal reasoning wrapped in <think>...</think> tags.

    Args:
        response: Raw generated response
        remove_thinking: Whether to remove <think>...</think> content

    Returns:
        Cleaned response content
    """
    if not response:
        return ""

    if remove_thinking:
        # Remove <think>...</think> blocks
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    return response.strip()
