#!/usr/bin/env python3
"""
Preprocess the insurance underwriting dataset.

This script loads the raw dataset, applies preprocessing, and saves the result.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_insurance_dataset
from src.data.preprocessing import preprocess_dataset, get_preprocessing_stats


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess insurance underwriting dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory to save preprocessed data",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / "data",
        help="Directory for dataset cache",
    )
    parser.add_argument(
        "--include-tool-calls",
        action="store_true",
        help="Include examples with tool calls",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of dataset",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Insurance Underwriting Dataset Preprocessing")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    try:
        dataset_dict = load_insurance_dataset(
            cache_dir=args.cache_dir,
            force_download=args.force_download,
        )
        dataset = dataset_dict["train"]
        print(f"   Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Preprocess dataset
    print("\n2. Preprocessing dataset...")
    try:
        preprocessed = preprocess_dataset(
            dataset,
            include_tool_calls=args.include_tool_calls,
            verbose=True,
        )
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return 1

    # Compute statistics
    print("\n3. Computing statistics...")
    stats = get_preprocessing_stats(preprocessed)
    print(f"   Preprocessed dataset statistics:")
    print(f"   - Examples: {stats['num_examples']}")
    print(f"   - Avg turns: {stats['turns']['mean']:.1f}")
    print(f"   - Turn range: {stats['turns']['min']}-{stats['turns']['max']}")
    print(f"   - Avg text length: {stats['text_length']['mean']:.0f} chars")

    # Save preprocessed data
    print("\n4. Saving preprocessed data...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save as HuggingFace dataset
        output_path = args.output_dir / "full"
        preprocessed.save_to_disk(str(output_path))
        print(f"   Saved to: {output_path}")

        # Save statistics
        stats_path = args.output_dir / "preprocessing_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"   Statistics saved to: {stats_path}")

    except Exception as e:
        print(f"Error saving data: {e}")
        return 1

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
