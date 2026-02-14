"""LoRA/QLoRA configuration for parameter-efficient finetuning."""

import logging

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from transformers import BitsAndBytesConfig, PreTrainedModel

from configs.model import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
)

logger = logging.getLogger(__name__)


def create_lora_config(
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """
    Create a LoRA configuration for causal language modeling.

    Args:
        r: LoRA rank (low-rank dimension)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to

    Returns:
        Configured LoraConfig

    Raises:
        ValueError: If r or lora_alpha are not positive
    """
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got {r}")
    if lora_alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {lora_dropout}")

    if target_modules is None:
        target_modules = DEFAULT_LORA_TARGET_MODULES

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    logger.info(
        f"Created LoRA config: r={r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, targets={target_modules}"
    )
    return config


def create_quantization_config() -> BitsAndBytesConfig:
    """
    Create a 4-bit quantization configuration for QLoRA.

    Uses NF4 quantization with double quantization for maximum memory savings.

    Returns:
        Configured BitsAndBytesConfig
    """
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Created QLoRA quantization config: NF4 with double quantization")
    return config


def apply_lora_adapters(
    model: PreTrainedModel,
    lora_config: LoraConfig,
) -> PeftModel:
    """
    Apply LoRA adapters to a base model.

    Freezes base model parameters, injects LoRA adapters, and enables
    gradient checkpointing for memory efficiency.

    Args:
        model: Base model to wrap with LoRA adapters
        lora_config: LoRA configuration

    Returns:
        PeftModel with LoRA adapters applied

    Raises:
        ValueError: If adapter injection fails
    """
    try:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

        peft_model = get_peft_model(model, lora_config)

        param_info = get_trainable_parameters(peft_model)
        logger.info(
            f"LoRA adapters applied: "
            f"{param_info['trainable_params']:,} trainable / "
            f"{param_info['total_params']:,} total "
            f"({param_info['trainable_percent']:.2f}%)"
        )

        return peft_model

    except Exception as e:
        raise ValueError(f"Failed to apply LoRA adapters: {e}") from e


def get_trainable_parameters(model: PeftModel | PreTrainedModel) -> dict[str, int | float]:
    """
    Report trainable vs total parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts:
            - trainable_params: Number of trainable parameters
            - total_params: Total number of parameters
            - trainable_percent: Percentage of trainable parameters
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percent = (trainable / total) * 100 if total > 0 else 0.0

    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": percent,
    }
