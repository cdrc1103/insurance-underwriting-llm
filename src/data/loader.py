"""Data loading utilities for insurance underwriting dataset."""

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def load_insurance_dataset(
    dataset_name: str = "snorkelai/Multi-Turn-Insurance-Underwriting",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> DatasetDict:
    """
    Load the insurance underwriting dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        cache_dir: Optional directory to cache the dataset
        force_download: If True, re-download even if cached

    Returns:
        DatasetDict containing the train split

    Raises:
        ValueError: If dataset cannot be loaded
    """
    try:
        if cache_dir and not force_download:
            cache_path = cache_dir / "raw" / "insurance_underwriting"
            if cache_path.exists():
                print(f"Loading dataset from cache: {cache_path}")
                return load_from_disk(str(cache_path))

        print(f"Downloading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name)

        # Cache dataset if cache_dir provided
        if cache_dir:
            cache_path = cache_dir / "raw" / "insurance_underwriting"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(cache_path))
            print(f"Dataset cached to: {cache_path}")

        return dataset

    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}': {e}") from e


def get_dataset_statistics(dataset: Dataset) -> dict[str, any]:
    """
    Compute basic statistics about the dataset.

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        "num_examples": len(dataset),
        "features": list(dataset.features.keys()),
        "column_names": dataset.column_names,
    }

    # Check for missing values
    if len(dataset) > 0:
        first_example = dataset[0]
        stats["example_keys"] = list(first_example.keys())

    return stats


def save_dataset_split(
    dataset: Dataset,
    output_path: Path,
    split_name: str = "train",
) -> None:
    """
    Save a dataset split to disk.

    Args:
        dataset: Dataset split to save
        output_path: Directory to save the dataset
        split_name: Name of the split (train/val/test)

    Raises:
        IOError: If dataset cannot be saved
    """
    try:
        save_path = output_path / split_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        print(f"Saved {split_name} split to: {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save dataset to {output_path}: {e}") from e


def load_dataset_split(
    input_path: Path,
    split_name: str = "train",
) -> Dataset:
    """
    Load a dataset split from disk.

    Args:
        input_path: Directory containing the dataset
        split_name: Name of the split to load (train/val/test)

    Returns:
        Loaded Dataset

    Raises:
        FileNotFoundError: If dataset split does not exist
        IOError: If dataset cannot be loaded
    """
    load_path = input_path / split_name

    if not load_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {load_path}")

    try:
        dataset = load_from_disk(str(load_path))
        print(f"Loaded {split_name} split from: {load_path}")
        return dataset
    except Exception as e:
        raise IOError(f"Failed to load dataset from {load_path}: {e}") from e
