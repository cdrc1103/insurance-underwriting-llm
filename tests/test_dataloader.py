"""Tests for data loading utilities."""

import pytest
import torch
from datasets import Dataset

from src.data.dataloader import (
    InsuranceConversationDataset,
    calculate_gradient_accumulation_steps,
    create_dataloader,
    estimate_memory_usage,
    get_batch_statistics,
)
from src.data.tokenization import load_tokenizer


@pytest.fixture
def sample_tokenized_dataset():
    """Create a sample tokenized dataset."""
    data = {
        "input_ids": [[1, 2, 3, 4, 5]] * 10,
        "attention_mask": [[1, 1, 1, 1, 1]] * 10,
        "labels": [[1, 2, 3, 4, 5]] * 10,
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_tokenizer():
    """Load a tokenizer for testing."""
    return load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")


def test_insurance_conversation_dataset(sample_tokenized_dataset):
    """Test PyTorch dataset wrapper."""
    dataset = InsuranceConversationDataset(sample_tokenized_dataset)

    assert len(dataset) == 10

    # Test getitem
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item

    # Check tensor types
    assert isinstance(item["input_ids"], torch.Tensor)
    assert item["input_ids"].dtype == torch.long


def test_insurance_conversation_dataset_with_metadata():
    """Test dataset with metadata."""
    data = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[1, 2, 3]],
        "original_index": [0],
        "num_turns": [2],
    }
    dataset = Dataset.from_dict(data)

    torch_dataset = InsuranceConversationDataset(dataset, return_metadata=True)
    item = torch_dataset[0]

    assert "original_index" in item
    assert "num_turns" in item
    assert item["original_index"] == 0
    assert item["num_turns"] == 2


def test_create_dataloader(sample_tokenized_dataset, sample_tokenizer):
    """Test dataloader creation."""
    dataloader = create_dataloader(
        sample_tokenized_dataset,
        sample_tokenizer,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    assert dataloader is not None
    assert len(dataloader) == 5  # 10 examples / batch_size 2

    # Test iteration
    batch = next(iter(dataloader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    # Check batch size
    assert batch["input_ids"].shape[0] == 2


def test_create_dataloader_invalid_dataset(sample_tokenizer):
    """Test dataloader with invalid dataset."""
    invalid_dataset = Dataset.from_dict({"text": ["hello", "world"]})

    with pytest.raises(ValueError, match="must have 'input_ids'"):
        create_dataloader(invalid_dataset, sample_tokenizer)


def test_get_batch_statistics():
    """Test batch statistics computation."""
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([[1, 2, 3, -100], [1, 2, -100, -100]]),
    }

    stats = get_batch_statistics(batch)

    assert stats["batch_size"] == 2
    assert stats["sequence_length"] == 4
    assert "actual_lengths" in stats
    assert stats["actual_lengths"]["mean"] == 2.5  # (3 + 2) / 2
    assert "label_ratio" in stats


def test_estimate_memory_usage():
    """Test memory usage estimation."""
    memory = estimate_memory_usage(
        batch_size=4,
        sequence_length=1024,
        model_params=1_500_000_000,  # Qwen2.5-1.5B
    )

    assert "model_gb" in memory
    assert "optimizer_gb" in memory
    assert "gradients_gb" in memory
    assert "activations_gb" in memory
    assert "total_gb" in memory

    # Check reasonable values
    assert memory["total_gb"] > 0
    assert memory["total_gb"] < 100  # Should be less than 100GB for Qwen2.5-1.5B


def test_calculate_gradient_accumulation_steps():
    """Test gradient accumulation calculation."""
    steps = calculate_gradient_accumulation_steps(
        effective_batch_size=32,
        per_device_batch_size=4,
        num_devices=1,
    )

    assert steps == 8  # 32 / (4 * 1)


def test_calculate_gradient_accumulation_steps_multi_gpu():
    """Test gradient accumulation with multiple GPUs."""
    steps = calculate_gradient_accumulation_steps(
        effective_batch_size=32,
        per_device_batch_size=4,
        num_devices=2,
    )

    assert steps == 4  # 32 / (4 * 2)


def test_calculate_gradient_accumulation_steps_invalid():
    """Test gradient accumulation with invalid parameters."""
    with pytest.raises(ValueError, match="Cannot achieve effective batch size"):
        calculate_gradient_accumulation_steps(
            effective_batch_size=4,
            per_device_batch_size=8,
            num_devices=1,
        )
