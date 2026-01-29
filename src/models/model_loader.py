"""Model loading utilities for base and fine-tuned models."""

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_base_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base Qwen3 model for inference.

    Args:
        model_name: HuggingFace model identifier
        device_map: Device placement strategy
        torch_dtype: Data type for model weights

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model cannot be loaded
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}") from e
