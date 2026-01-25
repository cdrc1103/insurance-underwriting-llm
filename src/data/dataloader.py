"""PyTorch data loading utilities for insurance underwriting conversations."""

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
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
