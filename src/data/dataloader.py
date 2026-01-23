"""PyTorch data loading utilities for insurance underwriting conversations."""

from typing import Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer


class InsuranceConversationDataset(TorchDataset):
    """
    PyTorch Dataset for insurance underwriting conversations.

    Args:
        dataset: HuggingFace Dataset with tokenized examples
        return_metadata: Whether to include metadata (original_index, num_turns)
    """

    def __init__(self, dataset: Dataset, return_metadata: bool = False):
        self.dataset = dataset
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single example.

        Args:
            idx: Example index

        Returns:
            Dictionary with tensors for model input
        """
        example = self.dataset[idx]

        # Convert to tensors
        item = {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
        }

        if "labels" in example:
            item["labels"] = torch.tensor(example["labels"], dtype=torch.long)

        # Add metadata if requested
        if self.return_metadata:
            if "original_index" in example:
                item["original_index"] = example["original_index"]
            if "num_turns" in example:
                item["num_turns"] = example["num_turns"]

        return item


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    mlm: bool = False,
    mlm_probability: float = 0.15,
) -> DataLoader:
    """
    Create a DataLoader for training or evaluation.

    Args:
        dataset: Tokenized HuggingFace Dataset
        tokenizer: Tokenizer used for the dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory (faster GPU transfer)
        mlm: Whether to use masked language modeling
        mlm_probability: Probability of masking tokens (if mlm=True)

    Returns:
        PyTorch DataLoader

    Raises:
        ValueError: If dataset is not properly tokenized
    """
    # Validate dataset
    required_fields = ["input_ids", "attention_mask"]
    for field in required_fields:
        if field not in dataset.column_names:
            raise ValueError(f"Dataset must have '{field}' column. Run tokenization first.")

    # Create PyTorch dataset
    torch_dataset = InsuranceConversationDataset(dataset)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
        mlm_probability=mlm_probability,
    )

    # Create dataloader
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=data_collator,
    )

    return dataloader


def get_batch_statistics(batch: dict[str, torch.Tensor]) -> dict:
    """
    Compute statistics for a batch.

    Args:
        batch: Batch dictionary from DataLoader

    Returns:
        Dictionary with batch statistics
    """
    stats = {
        "batch_size": batch["input_ids"].shape[0],
        "sequence_length": batch["input_ids"].shape[1],
    }

    if "attention_mask" in batch:
        # Compute actual lengths (non-padded)
        actual_lengths = batch["attention_mask"].sum(dim=1).tolist()
        stats["actual_lengths"] = {
            "mean": sum(actual_lengths) / len(actual_lengths),
            "min": min(actual_lengths),
            "max": max(actual_lengths),
        }

    if "labels" in batch:
        # Compute number of non-ignored labels (-100 is typically used for padding)
        non_ignored = (batch["labels"] != -100).sum().item()
        total = batch["labels"].numel()
        stats["label_ratio"] = non_ignored / total if total > 0 else 0

    return stats


def estimate_memory_usage(
    batch_size: int,
    sequence_length: int,
    vocab_size: int = 50257,  # GPT-2 vocab size
    model_params: int = 124_000_000,  # ~124M for GPT-2
    bytes_per_param: int = 4,  # float32
) -> dict[str, float]:
    """
    Estimate GPU memory usage for training.

    Args:
        batch_size: Training batch size
        sequence_length: Maximum sequence length
        vocab_size: Vocabulary size
        model_params: Number of model parameters
        bytes_per_param: Bytes per parameter (4 for float32, 2 for float16)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Model weights
    model_memory = model_params * bytes_per_param

    # Optimizer states (Adam: 2x parameters for momentum and variance)
    optimizer_memory = model_params * bytes_per_param * 2

    # Gradients (same size as parameters)
    gradient_memory = model_params * bytes_per_param

    # Activations (rough estimate: batch_size * sequence_length * hidden_size * layers * 4)
    # For GPT-2: hidden_size=768, layers=12
    activation_memory = batch_size * sequence_length * 768 * 12 * bytes_per_param * 4

    # Input batch
    input_memory = batch_size * sequence_length * bytes_per_param

    total_memory = (
        model_memory + optimizer_memory + gradient_memory + activation_memory + input_memory
    )

    return {
        "model_gb": model_memory / (1024**3),
        "optimizer_gb": optimizer_memory / (1024**3),
        "gradients_gb": gradient_memory / (1024**3),
        "activations_gb": activation_memory / (1024**3),
        "input_gb": input_memory / (1024**3),
        "total_gb": total_memory / (1024**3),
    }


def calculate_gradient_accumulation_steps(
    effective_batch_size: int,
    per_device_batch_size: int,
    num_devices: int = 1,
) -> int:
    """
    Calculate gradient accumulation steps to achieve effective batch size.

    Args:
        effective_batch_size: Desired effective batch size
        per_device_batch_size: Batch size per GPU
        num_devices: Number of GPUs

    Returns:
        Number of gradient accumulation steps

    Raises:
        ValueError: If calculation results in less than 1 step
    """
    steps = effective_batch_size // (per_device_batch_size * num_devices)

    if steps < 1:
        raise ValueError(
            f"Cannot achieve effective batch size {effective_batch_size} "
            f"with per_device_batch_size={per_device_batch_size} and "
            f"num_devices={num_devices}"
        )

    return steps
