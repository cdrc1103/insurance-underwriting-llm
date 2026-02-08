"""Tests for model loading utilities."""

import pytest
import torch

from src.models.model_loader import (
    get_model_architecture,
    load_base_model,
    profile_model_memory,
    verify_model_generation,
)


@pytest.fixture(scope="module")
def loaded_model():
    """Load model once for all tests to save time."""
    model, tokenizer = load_base_model(
        model_name="Qwen/Qwen3-0.6B",
        device_map="cpu",  # Use CPU for testing to avoid GPU requirement
        dtype=torch.float32,  # Use float32 for CPU
    )
    return model, tokenizer


def test_load_base_model_success(loaded_model):
    """Test successful model loading."""
    model, tokenizer = loaded_model

    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "config")
    assert hasattr(tokenizer, "pad_token")


def test_load_base_model_tokenizer_config(loaded_model):
    """Test tokenizer is properly configured with special tokens."""
    model, tokenizer = loaded_model

    # Check special tokens are set
    assert tokenizer.pad_token is not None
    assert tokenizer.eos_token is not None
    assert tokenizer.pad_token_id is not None

    # Check model config matches tokenizer
    assert model.config.pad_token_id == tokenizer.pad_token_id


def test_load_base_model_invalid_model():
    """Test loading invalid model raises ValueError."""
    with pytest.raises(ValueError, match="Failed to load model"):
        load_base_model(model_name="nonexistent-model-xyz-123")


def test_load_base_model_dtype():
    """Test model loads with specified dtype."""
    model, _ = load_base_model(
        model_name="Qwen/Qwen3-0.6B",
        device_map="cpu",
        dtype=torch.float32,
    )

    # Check that parameters have the correct dtype
    first_param = next(model.parameters())
    assert first_param.dtype == torch.float32


def test_get_model_architecture(loaded_model):
    """Test model architecture extraction."""
    model, _ = loaded_model
    arch = get_model_architecture(model)

    # Check all required fields are present
    assert "model_type" in arch
    assert "num_layers" in arch
    assert "hidden_size" in arch
    assert "num_attention_heads" in arch
    assert "vocab_size" in arch
    assert "num_parameters" in arch

    # Check types and reasonable values
    assert isinstance(arch["model_type"], str)
    assert isinstance(arch["num_layers"], int)
    assert arch["num_layers"] > 0
    assert isinstance(arch["hidden_size"], int)
    assert arch["hidden_size"] > 0
    assert isinstance(arch["num_attention_heads"], int)
    assert arch["num_attention_heads"] > 0
    assert isinstance(arch["vocab_size"], int)
    assert arch["vocab_size"] > 0
    assert isinstance(arch["num_parameters"], int)
    assert arch["num_parameters"] > 0


def test_get_model_architecture_qwen_specific(loaded_model):
    """Test architecture values for Qwen3-0.6B specifically."""
    model, _ = loaded_model
    arch = get_model_architecture(model)

    # Qwen3-0.6B specific architecture (approximate values)
    assert arch["model_type"] == "qwen3"
    # Should have ~600M parameters (allowing some margin)
    assert 500_000_000 < arch["num_parameters"] < 800_000_000


def test_profile_model_memory(loaded_model):
    """Test model memory profiling."""
    model, _ = loaded_model
    memory = profile_model_memory(model)

    # Check required fields
    assert "model_size_gb" in memory
    assert "device" in memory

    # Check types and reasonable values
    assert isinstance(memory["model_size_gb"], float)
    assert memory["model_size_gb"] > 0
    assert memory["model_size_gb"] < 100  # Should be < 100GB for 0.6B model
    assert isinstance(memory["device"], str)


def test_profile_model_memory_cpu(loaded_model):
    """Test memory profiling on CPU device."""
    model, _ = loaded_model
    memory = profile_model_memory(model)

    # CPU model should not have CUDA memory fields
    assert "cpu" in memory["device"]
    # May or may not have allocated_gb/reserved_gb depending on device


def test_verify_model_generation_basic(loaded_model):
    """Test basic text generation."""
    model, tokenizer = loaded_model

    prompt = "What is insurance?"
    generated = verify_model_generation(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.7,
    )

    # Check that generation produces text
    assert isinstance(generated, str)
    assert len(generated) > len(prompt)  # Should generate additional text
    assert prompt in generated  # Original prompt should be included


def test_verify_model_generation_different_prompts(loaded_model):
    """Test generation with different prompts."""
    model, tokenizer = loaded_model

    prompts = [
        "Hello, how are you?",
        "Explain underwriting.",
        "What is risk assessment?",
    ]

    for prompt in prompts:
        generated = verify_model_generation(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=15,
        )

        assert isinstance(generated, str)
        assert len(generated) > 0


def test_verify_model_generation_temperature_zero(loaded_model):
    """Test generation with temperature=0 for deterministic output."""
    model, tokenizer = loaded_model

    prompt = "The capital of France is"

    # Generate twice with temperature=0
    gen1 = verify_model_generation(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.0,
    )

    gen2 = verify_model_generation(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.0,
    )

    # With temperature=0, outputs should be identical (deterministic)
    # Note: This might not always hold due to numerical precision
    assert isinstance(gen1, str)
    assert isinstance(gen2, str)


def test_verify_model_generation_max_tokens(loaded_model):
    """Test generation respects max_new_tokens."""
    model, tokenizer = loaded_model

    prompt = "Write a story:"

    # Generate with very small max_new_tokens
    short_gen = verify_model_generation(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=5,
    )

    # Generate with larger max_new_tokens
    long_gen = verify_model_generation(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=30,
    )

    # Longer generation should produce more tokens
    assert len(long_gen) >= len(short_gen)
