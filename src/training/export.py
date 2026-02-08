"""Model adapter export, loading, and versioning."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def export_adapter(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str | Path,
    training_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """
    Export LoRA adapter weights and metadata.

    Saves adapter weights, tokenizer, training configuration, and
    metadata for reproducibility.

    Args:
        model: Trained PeftModel with LoRA adapters
        tokenizer: Tokenizer
        output_dir: Directory to save adapter files
        training_config: Training arguments dictionary
        metrics: Training metrics (loss, perplexity, etc.)

    Returns:
        Path to the output directory

    Raises:
        ValueError: If model is not a PeftModel
    """
    if not isinstance(model, PeftModel):
        raise ValueError("Model must be a PeftModel with LoRA adapters")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter weights
    model.save_pretrained(output_dir)
    logger.info(f"Adapter weights saved to {output_dir}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")

    # Save training config
    if training_config is not None:
        config_path = output_dir / "training_args.json"
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2, default=str)
        logger.info(f"Training config saved to {config_path}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "base_model": model.peft_config["default"].base_model_name_or_path,
        "lora_r": model.peft_config["default"].r,
        "lora_alpha": model.peft_config["default"].lora_alpha,
    }
    if metrics is not None:
        metadata["metrics"] = metrics

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")

    return output_dir


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str | Path,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """
    Load a base model with finetuned LoRA adapters.

    Args:
        base_model_name: HuggingFace model identifier for the base model
        adapter_path: Path to saved LoRA adapter directory
        device_map: Device placement strategy
        dtype: Data type for model weights

    Returns:
        Tuple of (PeftModel with adapters, tokenizer)

    Raises:
        ValueError: If adapter path doesn't exist or loading fails
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise ValueError(f"Adapter path does not exist: {adapter_path}")

    try:
        from transformers import AutoTokenizer

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )

        # Load adapter
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

        # Load tokenizer from adapter dir (has same special tokens config)
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        logger.info(f"Loaded finetuned model from {adapter_path}")
        return model, tokenizer

    except Exception as e:
        raise ValueError(f"Failed to load finetuned model: {e}") from e


def merge_adapter(model: PeftModel) -> PreTrainedModel:
    """
    Merge LoRA adapter weights into the base model.

    Creates a standalone model without adapter overhead for deployment.

    Args:
        model: PeftModel with LoRA adapters

    Returns:
        Base model with merged adapter weights

    Raises:
        ValueError: If model is not a PeftModel
    """
    if not isinstance(model, PeftModel):
        raise ValueError("Model must be a PeftModel with LoRA adapters")

    merged = model.merge_and_unload()
    logger.info("LoRA adapters merged into base model")
    return merged
