"""Inference pipeline for generating model responses."""

from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate response from model given conversation messages.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        messages: List of message dicts with 'role' and 'content' keys
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens

    Returns:
        Generated response text

    Raises:
        ValueError: If generation fails
    """
    try:
        # Format messages using chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode response (skip the input prompt)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()

    except Exception as e:
        raise ValueError(f"Generation failed: {e}") from e


def evaluate_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Any,
    generation_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate responses for all examples in dataset.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        dataset: Dataset with 'messages' field
        generation_kwargs: Optional generation parameters
        verbose: Whether to print progress

    Returns:
        List of dicts with original example + generated response

    Raises:
        ValueError: If dataset is invalid
    """
    if generation_kwargs is None:
        generation_kwargs = {}

    results = []

    for i, example in enumerate(dataset):
        if verbose and i % 10 == 0:
            print(f"Processing example {i}/{len(dataset)}...")

        try:
            response = generate_response(model, tokenizer, example["messages"], **generation_kwargs)

            results.append(
                {
                    "original_index": example.get("original_index", i),
                    "task": example.get("task", "unknown"),
                    "messages": example["messages"],
                    "reference_answer": example.get("reference_answer", ""),
                    "generated_response": response,
                }
            )

        except Exception as e:
            print(f"Failed to generate response for example {i}: {e}")
            results.append(
                {
                    "original_index": example.get("original_index", i),
                    "task": example.get("task", "unknown"),
                    "messages": example["messages"],
                    "reference_answer": example.get("reference_answer", ""),
                    "generated_response": f"ERROR: {str(e)}",
                }
            )

    if verbose:
        print(f"Completed evaluation on {len(results)} examples")

    return results
