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


class StopTokensCriteria(StoppingCriteria):
    """
    Stopping criteria that halts text generation when a stop *token sequence*
    is encountered.

    This implementation pre-tokenizes each stop string and compares the most
    recently generated tokens against those token sequences. By operating at
    the token level, it avoids repeatedly decoding text and scanning strings,
    making it significantly more efficient than text-based approaches.

    Key advantages:
    - No repeated decoding of generated tokens
    - Constant-time checks per generation step
    - Exact matching that correctly handles token boundaries
    - Scales well for long generations
    """

    def __init__(
        self,
        stop_strings: list[str],
        tokenizer: PreTrainedTokenizer,
        input_length: int,
    ) -> None:
        """
        Initialize stop strings criteria.

        Args:
            stop_strings: List of strings that trigger stopping
            tokenizer: Tokenizer for decoding generated tokens
            input_length: Length of input tokens to skip when checking
        """
        self.stop_token_seqs = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
        self.input_length = input_length

    def __call__(
        self, input_ids: torch.LongTensor, _scores: torch.FloatTensor, **_kwargs: Any
    ) -> bool:
        """Check if any stop string is in generated text."""
        generated = input_ids[0, self.input_length :]

        for stop_seq in self.stop_token_seqs:
            if len(generated) >= len(stop_seq):
                if generated[-len(stop_seq) :].tolist() == stop_seq:
                    return True
        return False


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    config: GenerationConfig | None = None,
) -> str:
    """
    Generate response from model given conversation messages.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        messages: List of message dicts with 'role' and 'content' keys
        config: Generation configuration (uses defaults if not provided)

    Returns:
        Generated response text

    Raises:
        ValueError: If generation fails
    """
    if config is None:
        config = GenerationConfig()

    try:
        # Format messages using chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        # Build stopping criteria
        stopping_criteria = None
        if config.stop_strings:
            stopping_criteria = StoppingCriteriaList(
                [StopTokensCriteria(config.stop_strings, tokenizer, input_length)]
            )

        # Generate
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

        # Decode response (skip the input prompt)
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        # Trim response at stop strings if present
        response = _trim_at_stop_strings(response, config.stop_strings)

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


def _process_single_example(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    example: dict[str, Any],
    index: int,
    config: GenerationConfig,
) -> tuple[dict[str, Any], bool]:
    """
    Process a single example and return result dict and success flag.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        example: Example dict with 'messages' field
        index: Index for tracking in results
        config: Generation configuration

    Returns:
        Tuple of (result_dict, success_flag)
    """
    if "messages" not in example:
        raise ValueError(f"Example {index} missing required 'messages' field")

    try:
        gen_result = generate_response_with_metadata(
            model, tokenizer, example["messages"], config=config
        )

        result = {
            "original_index": example.get("original_index", index),
            "task": example.get("task", "unknown"),
            "messages": example["messages"],
            "reference_answer": example.get("reference_answer", ""),
            "generated_response": gen_result.response,
            "generation_time_ms": gen_result.generation_time_ms,
            "input_tokens": gen_result.input_tokens,
            "output_tokens": gen_result.output_tokens,
        }
        return result, True

    except Exception as e:
        result = {
            "original_index": example.get("original_index", index),
            "task": example.get("task", "unknown"),
            "messages": example["messages"],
            "reference_answer": example.get("reference_answer", ""),
            "generated_response": None,
            "error": str(e),
            "generation_time_ms": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        return result, False


def evaluate_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Sequence[dict[str, Any]],
    config: GenerationConfig | None = None,
) -> EvaluationResult:
    """
    Generate responses for all examples in dataset.

    Processes examples sequentially with detailed metadata tracking.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        dataset: Dataset with 'messages' field
        config: Generation configuration (uses defaults if not provided)

    Returns:
        EvaluationResult with all responses and metadata

    Raises:
        ValueError: If dataset is empty or missing 'messages' field
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if config is None:
        config = GenerationConfig()

    results = []
    successful_count = 0
    failed_count = 0

    start_time = time.perf_counter()

    for i, example in enumerate(dataset):
        result, success = _process_single_example(model, tokenizer, example, i, config)
        results.append(result)
        if success:
            successful_count += 1
        else:
            failed_count += 1

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

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

    Limitation: Stop string criteria uses the padded sequence length for all
    sequences in the batch. For batches with highly variable sequence lengths,
    this may cause stop strings to check against padding tokens rather than
    only generated content. Post-processing trims stop strings from the decoded
    text to mitigate this.

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
        # For batched generation, we use the padded sequence length as reference.
        # This means the stopping criteria may check against padding tokens for
        # shorter sequences. Post-processing trims stop strings to handle this.
        stopping_criteria = StoppingCriteriaList(
            [StopTokensCriteria(config.stop_strings, tokenizer, inputs["input_ids"].shape[1])]
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

    # Batch decode all responses
    padded_input_len = inputs["input_ids"].shape[1]
    generated_only = [output_ids[padded_input_len:] for output_ids in outputs]
    responses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    # Build results from decoded responses
    results = []
    for response, input_len, generated_ids in zip(
        responses, input_lengths, generated_only, strict=True
    ):
        # Trim at stop strings
        response = _trim_at_stop_strings(response, config.stop_strings)
        response = response.strip()

        output_tokens = len(generated_ids)

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
) -> EvaluationResult:
    """
    Generate responses for dataset using batched inference for efficiency.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        dataset: Dataset with 'messages' field
        config: Generation configuration
        batch_size: Number of examples to process in each batch

    Returns:
        EvaluationResult with all responses and metadata

    Raises:
        ValueError: If dataset is empty or missing required fields
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if config is None:
        config = GenerationConfig()

    results = []
    successful_count = 0
    failed_count = 0

    start_time = time.perf_counter()

    # Process in batches
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_examples = [dataset[i] for i in range(batch_start, batch_end)]

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

        except Exception as batch_error:
            # Fall back to sequential for this batch
            logger.warning(
                f"Batch processing failed for batch starting at index {batch_start}: {batch_error}. "
                f"Falling back to sequential processing for this batch."
            )
            for i, example in enumerate(batch_examples):
                result, success = _process_single_example(
                    model, tokenizer, example, batch_start + i, config
                )
                results.append(result)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

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
