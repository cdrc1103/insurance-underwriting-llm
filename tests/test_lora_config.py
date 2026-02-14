"""Tests for LoRA/QLoRA configuration."""

import pytest
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.training.lora_config import (
    apply_lora_adapters,
    create_lora_config,
    create_quantization_config,
    get_trainable_parameters,
)


@pytest.fixture(scope="module")
def base_model():
    """Load a small model for testing."""
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return model


class TestCreateLoraConfig:
    """Tests for create_lora_config."""

    def test_default_config(self):
        """Test creating config with default parameters."""
        config = create_lora_config()
        assert isinstance(config, LoraConfig)
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.task_type == "CAUSAL_LM"

    def test_custom_rank(self):
        """Test creating config with custom rank."""
        config = create_lora_config(r=8, lora_alpha=16)
        assert config.r == 8
        assert config.lora_alpha == 16

    def test_custom_target_modules(self):
        """Test creating config with custom target modules."""
        targets = ["q_proj", "v_proj"]
        config = create_lora_config(target_modules=targets)
        assert config.target_modules == set(targets)

    def test_invalid_rank_raises(self):
        """Test that non-positive rank raises ValueError."""
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            create_lora_config(r=0)

    def test_invalid_alpha_raises(self):
        """Test that non-positive alpha raises ValueError."""
        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            create_lora_config(lora_alpha=-1)

    def test_invalid_dropout_raises(self):
        """Test that invalid dropout raises ValueError."""
        with pytest.raises(ValueError, match="LoRA dropout must be in"):
            create_lora_config(lora_dropout=1.0)


class TestCreateQuantizationConfig:
    """Tests for create_quantization_config."""

    def test_creates_4bit_config(self):
        """Test that quantization config uses 4-bit NF4."""
        config = create_quantization_config()
        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True


class TestApplyLoraAdapters:
    """Tests for apply_lora_adapters."""

    def test_adapters_applied(self, base_model):
        """Test that LoRA adapters are injected."""
        lora_config = create_lora_config(r=8, lora_alpha=16)
        peft_model = apply_lora_adapters(base_model, lora_config)

        # Should have peft config
        assert hasattr(peft_model, "peft_config")
        assert "default" in peft_model.peft_config

    def test_trainable_params_small_fraction(self, base_model):
        """Test that trainable params are < 5% of total."""
        lora_config = create_lora_config(r=8, lora_alpha=16)
        peft_model = apply_lora_adapters(base_model, lora_config)

        info = get_trainable_parameters(peft_model)
        assert info["trainable_percent"] < 5.0
        assert info["trainable_params"] > 0
        assert info["total_params"] > info["trainable_params"]

    def test_base_params_frozen(self, base_model):
        """Test that base model parameters are frozen."""
        lora_config = create_lora_config(r=8, lora_alpha=16)
        peft_model = apply_lora_adapters(base_model, lora_config)

        # Check that non-LoRA params are frozen
        for name, param in peft_model.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, f"Parameter {name} should be frozen"


class TestGetTrainableParameters:
    """Tests for get_trainable_parameters."""

    def test_reports_correct_format(self, base_model):
        """Test that output has expected keys."""
        info = get_trainable_parameters(base_model)
        assert "trainable_params" in info
        assert "total_params" in info
        assert "trainable_percent" in info
        assert isinstance(info["trainable_params"], int)
        assert isinstance(info["total_params"], int)
        assert isinstance(info["trainable_percent"], float)
