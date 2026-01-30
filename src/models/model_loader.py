"""Model loading utilities for base and fine-tuned models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_base_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base Qwen3 model for inference.

    Args:
        model_name: HuggingFace model identifier
        device_map: Device placement strategy ("auto", "cuda", "cpu")
        torch_dtype: Data type for model weights (torch.float16, torch.bfloat16, or torch.float32)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model cannot be loaded
        RuntimeError: If GPU is requested but not available
    """
    # Validate device availability
    if device_map == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        # Configure special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Ensure pad_token_id is set in model config
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}") from e


def get_model_architecture(model: PreTrainedModel) -> dict[str, int | str]:
    """
    Extract model architecture details.

    Args:
        model: Loaded transformer model

    Returns:
        Dictionary with architecture information:
            - model_type: Model architecture type
            - num_layers: Number of transformer layers
            - hidden_size: Hidden dimension size
            - num_attention_heads: Number of attention heads
            - vocab_size: Tokenizer vocabulary size
            - num_parameters: Total number of parameters
    """
    config = model.config

    num_params = sum(p.numel() for p in model.parameters())

    return {
        "model_type": config.model_type,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "num_parameters": num_params,
    }


def profile_model_memory(model: PreTrainedModel) -> dict[str, float]:
    """
    Profile model memory usage.

    Args:
        model: Loaded transformer model

    Returns:
        Dictionary with memory information in GB:
            - model_size_gb: Model size in VRAM/RAM
            - allocated_gb: Total allocated memory (if CUDA)
            - reserved_gb: Total reserved memory (if CUDA)
            - device: Device where model is loaded

    Raises:
        RuntimeError: If memory profiling fails
    """
    try:
        # Calculate model size
        model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_gb = model_size_bytes / (1024**3)

        device = next(model.parameters()).device

        memory_info = {
            "model_size_gb": round(model_size_gb, 3),
            "device": str(device),
        }

        # Add CUDA memory stats if available
        if device.type == "cuda":
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            memory_info["allocated_gb"] = round(allocated_gb, 3)
            memory_info["reserved_gb"] = round(reserved_gb, 3)

        return memory_info

    except Exception as e:
        raise RuntimeError(f"Failed to profile model memory: {e}") from e


def verify_model_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str = "What is insurance underwriting?",
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """
    Verify model text generation on a sample prompt.

    Args:
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (use 0.0 for greedy decoding)

    Returns:
        Generated text response

    Raises:
        ValueError: If generation fails
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use greedy decoding for temperature=0, sampling otherwise
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if temperature == 0.0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        raise ValueError(f"Generation failed: {e}") from e
