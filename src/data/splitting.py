"""Dataset splitting utilities with stratification support."""

import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def create_stratified_split(
    dataset: Dataset,
    train_size: float = 0.75,
    val_size: float = 0.125,
    test_size: float = 0.125,
    stratify_by: str | None = "task",
    random_seed: int = 42,
) -> DatasetDict:
    """
    Create stratified train/val/test splits.

    Args:
        dataset: Dataset to split
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        stratify_by: Field to stratify by (None for random split, default: "task")
        random_seed: Random seed for reproducibility

    Returns:
        DatasetDict with train, validation, and test splits

    Raises:
        ValueError: If split sizes don't sum to 1.0 or stratify_by field is missing
    """
    # Validate split sizes
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")

    # Validate stratification field exists
    if stratify_by and stratify_by not in dataset.column_names:
        raise ValueError(
            f"Stratification field '{stratify_by}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    # Get indices and stratification labels
    indices = list(range(len(dataset)))
    stratify_labels = None

    if stratify_by and stratify_by in dataset.column_names:
        stratify_labels = dataset[stratify_by]

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )

    # Second split: val vs test
    # Adjust sizes for the remaining portion
    val_ratio = val_size / (val_size + test_size)

    if stratify_by and stratify_labels:
        temp_stratify = [stratify_labels[i] for i in temp_idx]
    else:
        temp_stratify = None

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_ratio,
        random_state=random_seed,
        stratify=temp_stratify,
    )

    # Create split datasets
    splits = {
        "train": dataset.select(train_idx),
        "validation": dataset.select(val_idx),
        "test": dataset.select(test_idx),
    }

    return DatasetDict(splits)


def compute_split_statistics(split: Dataset, split_name: str) -> dict:
    """
    Compute statistics for a dataset split.

    Args:
        split: Dataset split
        split_name: Name of the split (train/validation/test)

    Returns:
        Dictionary containing split statistics
    """
    stats = {
        "split_name": split_name,
        "num_examples": len(split),
    }

    # Task distribution
    if "task" in split.column_names:
        tasks = split["task"]
        task_distribution = {}
        for task in set(tasks):
            count = tasks.count(task)
            task_distribution[task] = {
                "count": count,
                "percentage": count / len(tasks) * 100,
            }
        stats["task_distribution"] = task_distribution

    # Conversation length statistics
    if "num_turns" in split.column_names:
        num_turns = split["num_turns"]
        stats["conversation_length"] = {
            "mean": float(np.mean(num_turns)),
            "median": float(np.median(num_turns)),
            "min": int(np.min(num_turns)),
            "max": int(np.max(num_turns)),
            "std": float(np.std(num_turns)),
        }

    # Text length statistics
    if "formatted_text" in split.column_names:
        text_lengths = [len(text) for text in split["formatted_text"]]
        stats["text_length"] = {
            "mean": float(np.mean(text_lengths)),
            "median": float(np.median(text_lengths)),
            "min": int(np.min(text_lengths)),
            "max": int(np.max(text_lengths)),
            "std": float(np.std(text_lengths)),
        }

    return stats


def save_splits(
    splits: DatasetDict,
    output_dir: Path,
    save_stats: bool = True,
) -> None:
    """
    Save dataset splits to disk.

    Args:
        splits: DatasetDict containing splits
        output_dir: Directory to save splits
        save_stats: Whether to save statistics file

    Raises:
        IOError: If splits cannot be saved
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        for split_name, split_data in splits.items():
            split_path = output_dir / split_name
            split_data.save_to_disk(str(split_path))
            print(f"Saved {split_name} split ({len(split_data)} examples) to: {split_path}")

        # Save statistics
        if save_stats:
            all_stats = {}
            for split_name, split_data in splits.items():
                all_stats[split_name] = compute_split_statistics(split_data, split_name)

            stats_path = output_dir / "split_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(all_stats, f, indent=2)
            print(f"Saved statistics to: {stats_path}")

    except Exception as e:
        raise OSError(f"Failed to save splits: {e}") from e


def load_splits(input_dir: Path) -> DatasetDict:
    """
    Load dataset splits from disk.

    Args:
        input_dir: Directory containing splits

    Returns:
        DatasetDict with loaded splits

    Raises:
        FileNotFoundError: If splits directory does not exist
        IOError: If splits cannot be loaded
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {input_dir}")

    try:
        splits = {}
        for split_name in ["train", "validation", "test"]:
            split_path = input_dir / split_name
            if split_path.exists():
                from datasets import load_from_disk

                splits[split_name] = load_from_disk(str(split_path))
                print(f"Loaded {split_name} split ({len(splits[split_name])} examples)")

        if not splits:
            raise OSError("No splits found in directory")

        return DatasetDict(splits)

    except Exception as e:
        raise OSError(f"Failed to load splits: {e}") from e


def print_split_summary(splits: DatasetDict) -> None:
    """
    Print a summary of dataset splits.

    Args:
        splits: DatasetDict containing splits
    """
    print("\n" + "=" * 80)
    print("Dataset Split Summary")
    print("=" * 80)

    total_examples = sum(len(split) for split in splits.values())

    for split_name, split_data in splits.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Examples: {len(split_data)} ({len(split_data) / total_examples * 100:.1f}%)")

        if "task" in split_data.column_names:
            tasks = split_data["task"]
            print("  Task distribution:")
            for task in sorted(set(tasks)):
                count = tasks.count(task)
                print(f"    - {task}: {count} ({count / len(tasks) * 100:.1f}%)")

        if "num_turns" in split_data.column_names:
            turns = split_data["num_turns"]
            print(f"  Conversation turns: {np.mean(turns):.1f} ± {np.std(turns):.1f}")

    print("\n" + "=" * 80)
