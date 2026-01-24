#!/usr/bin/env python3
"""
Create train/validation/test splits from preprocessed dataset.

This script loads preprocessed data and creates stratified splits.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_io import load_dataset_split
from src.data.splitting import create_stratified_split, print_split_summary, save_splits


def main():
    """Main splitting function."""
    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "splits",
        help="Directory to save splits",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.75,
        help="Proportion for training set (default: 0.75)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.125,
        help="Proportion for validation set (default: 0.125)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.125,
        help="Proportion for test set (default: 0.125)",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="task_type",
        help="Field to stratify by (default: task_type)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Dataset Splitting")
    print("=" * 80)

    # Validate split sizes
    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 0.001:
        print(f"Error: Split sizes must sum to 1.0, got {total}")
        return 1

    # Load preprocessed data
    print("\n1. Loading preprocessed dataset...")
    try:
        dataset = load_dataset_split(args.input_dir, split_name="full")
        print(f"   Loaded {len(dataset)} examples")
    except FileNotFoundError:
        print(f"Error: Preprocessed data not found at {args.input_dir}")
        print("Run preprocess_data.py first to create preprocessed data")
        return 1
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Create splits
    print("\n2. Creating stratified splits...")
    print(f"   Train: {args.train_size * 100:.1f}%")
    print(f"   Validation: {args.val_size * 100:.1f}%")
    print(f"   Test: {args.test_size * 100:.1f}%")
    print(f"   Stratify by: {args.stratify_by}")
    print(f"   Random seed: {args.random_seed}")

    try:
        splits = create_stratified_split(
            dataset,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            stratify_by=args.stratify_by,
            random_seed=args.random_seed,
        )
    except Exception as e:
        print(f"Error creating splits: {e}")
        return 1

    # Print summary
    print_split_summary(splits)

    # Save splits
    print("\n3. Saving splits...")
    try:
        save_splits(splits, args.output_dir, save_stats=True)
    except Exception as e:
        print(f"Error saving splits: {e}")
        return 1

    print("\n" + "=" * 80)
    print("Splitting complete!")
    print(f"Splits saved to: {args.output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
