#!/usr/bin/env python3
"""Token analysis script for insurance underwriting dataset.

This script analyzes token distributions across train/val/test splits and
determines the optimal max_length for training based on the 95th percentile.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.data.splitting import load_splits
from src.data.tokenization import (
    compute_token_statistics,
    get_recommended_max_length,
    load_tokenizer,
    mark_truncated_examples,
)


def analyze_all_splits(
    splits_dir: Path,
    model_name: str = "Qwen/Qwen3-0.6B",
    percentile: float = 95.0,
    output_file: Path | None = None,
) -> dict:
    """
    Analyze token statistics for all dataset splits.

    Args:
        splits_dir: Directory containing train/val/test splits
        model_name: Model name for tokenizer
        percentile: Percentile for max_length recommendation
        output_file: Optional path to save results JSON

    Returns:
        Dictionary with comprehensive token analysis results
    """
    print("=" * 80)
    print("Token Analysis for Insurance Underwriting Dataset")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Splits directory: {splits_dir}")
    print(f"Recommendation percentile: {percentile}th")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print()

    # Load splits
    print("Loading dataset splits...")
    splits = load_splits(splits_dir)
    print()

    # Analyze each split
    all_stats = {}

    for split_name in ["train", "validation", "test"]:
        if split_name not in splits:
            print(f"Warning: {split_name} split not found, skipping...")
            continue

        split = splits[split_name]
        print(f"Analyzing {split_name.upper()} split ({len(split)} examples)...")
        print("-" * 80)

        # Compute token statistics
        stats = compute_token_statistics(split, tokenizer)

        # Print detailed statistics
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Mean tokens: {stats['mean']:.1f}")
        print(f"  Median tokens: {stats['median']:.1f}")
        print(f"  Min tokens: {stats['min']}")
        print(f"  Max tokens: {stats['max']}")
        print(f"  Std deviation: {stats['std']:.1f}")
        print()
        print("  Percentiles:")
        for pct, value in stats["percentiles"].items():
            print(f"    {pct}: {value:.1f}")
        print()

        all_stats[split_name] = stats

    # Determine recommended max_length based on train split
    print("=" * 80)
    print("Recommendation Analysis")
    print("=" * 80)

    train_split = splits["train"]
    recommended_max_length = get_recommended_max_length(
        train_split, tokenizer, percentile=percentile
    )

    print(
        f"\nRecommended max_length (based on {percentile}th percentile): {recommended_max_length}"
    )
    print()

    # Perform truncation analysis for all splits at recommended max_length
    print("=" * 80)
    print("Truncation Analysis")
    print("=" * 80)
    print(f"Analyzing truncation at max_length={recommended_max_length}")
    print()

    truncation_stats = {}

    for split_name in ["train", "validation", "test"]:
        if split_name not in splits:
            continue

        split = splits[split_name]
        print(f"{split_name.upper()} split:")

        # Analyze truncation
        _, trunc_stats = mark_truncated_examples(
            split, tokenizer, max_length=recommended_max_length, verbose=False
        )

        # Print truncation statistics
        print(f"  Total examples: {trunc_stats['total_examples']}")
        print(
            f"  Will be truncated: {trunc_stats['truncated_count']} "
            f"({trunc_stats['truncated_percentage']:.1f}%)"
        )

        if trunc_stats["truncated_count"] > 0:
            print(f"  Max tokens over limit: {trunc_stats['max_tokens_over']}")
            print(f"  Avg tokens over limit: {trunc_stats['avg_tokens_over']:.0f}")
            print(f"  Truncated indices: {trunc_stats['truncated_indices']}")

        print()
        truncation_stats[split_name] = trunc_stats

    # Compile final results
    results = {
        "model_name": model_name,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "recommendation_percentile": percentile,
        "recommended_max_length": recommended_max_length,
        "token_statistics": all_stats,
        "truncation_analysis": truncation_stats,
    }

    # Save results if output file specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
        print()

    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Recommended max_length for training: {recommended_max_length} tokens")
    print(f"This covers {percentile}% of examples in the training set without truncation.")

    total_truncated = sum(s["truncated_count"] for s in truncation_stats.values())
    total_examples = sum(s["total_examples"] for s in truncation_stats.values())

    print("\nOverall truncation impact:")
    print(
        f"  {total_truncated}/{total_examples} examples "
        f"({total_truncated / total_examples * 100:.1f}%) will be truncated"
    )
    print()

    return results


def main():
    """Main entry point for token analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze token distributions and recommend max_length for training"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name for tokenizer",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile for max_length recommendation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/token_analysis.json"),
        help="Output file for analysis results",
    )

    args = parser.parse_args()

    # Run analysis
    analyze_all_splits(
        splits_dir=args.splits_dir,
        model_name=args.model_name,
        percentile=args.percentile,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
