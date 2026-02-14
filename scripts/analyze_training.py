#!/usr/bin/env python3
"""Analyze training convergence from TensorBoard logs.

Reads training logs and generates convergence analysis plots
including loss curves, learning rate schedule, and overfitting detection.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_tensorboard_events(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """
    Load metrics from TensorBoard event files.

    Args:
        log_dir: Directory containing TensorBoard event files

    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples

    Raises:
        ValueError: If no event files found
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        raise ImportError("tensorboard is required. Install with: pip install tensorboard") from e

    # Find event files recursively
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")

    metrics: dict[str, list[tuple[int, float]]] = {}

    for event_file in event_files:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()

        for tag in ea.Tags().get("scalars", []):
            if tag not in metrics:
                metrics[tag] = []
            for event in ea.Scalars(tag):
                metrics[tag].append((event.step, event.value))

    # Sort by step
    for tag in metrics:
        metrics[tag].sort(key=lambda x: x[0])

    logger.info(f"Loaded {len(metrics)} metrics from {len(event_files)} event files")
    return metrics


def smooth_values(values: list[float], window: int = 5) -> list[float]:
    """
    Apply simple moving average smoothing.

    Args:
        values: Raw values to smooth
        window: Smoothing window size

    Returns:
        Smoothed values
    """
    if len(values) <= window:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    # Pad beginning with original values
    pad = values[: len(values) - len(smoothed)]
    return pad + smoothed.tolist()


def plot_training_curves(
    metrics: dict[str, list[tuple[int, float]]],
    output_dir: Path,
) -> None:
    """
    Generate training convergence plots.

    Args:
        metrics: Dictionary of metric name to (step, value) tuples
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Convergence Analysis", fontsize=14)

    # Plot 1: Training loss
    ax = axes[0, 0]
    if "train/loss" in metrics:
        steps, values = zip(*metrics["train/loss"], strict=True)
        ax.plot(steps, values, alpha=0.3, color="blue", label="Raw")
        smoothed = smooth_values(list(values))
        ax.plot(steps[: len(smoothed)], smoothed, color="blue", label="Smoothed")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Train vs Validation loss
    ax = axes[0, 1]
    if "train/loss" in metrics:
        steps, values = zip(*metrics["train/loss"], strict=True)
        smoothed = smooth_values(list(values))
        ax.plot(steps[: len(smoothed)], smoothed, color="blue", label="Train")
    if "eval/loss" in metrics:
        steps, values = zip(*metrics["eval/loss"], strict=True)
        ax.plot(steps, values, color="orange", marker="o", label="Validation")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Learning rate
    ax = axes[1, 0]
    if "train/learning_rate" in metrics:
        steps, values = zip(*metrics["train/learning_rate"], strict=True)
        ax.plot(steps, values, color="green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    # Plot 4: Gradient norm
    ax = axes[1, 1]
    if "train/grad_norm" in metrics:
        steps, values = zip(*metrics["train/grad_norm"], strict=True)
        ax.plot(steps, values, alpha=0.3, color="red", label="Raw")
        smoothed = smooth_values(list(values))
        ax.plot(steps[: len(smoothed)], smoothed, color="red", label="Smoothed")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {plot_path}")


def generate_convergence_report(
    metrics: dict[str, list[tuple[int, float]]],
    output_dir: Path,
) -> dict:
    """
    Generate a convergence analysis report.

    Args:
        metrics: Dictionary of metric name to (step, value) tuples
        output_dir: Directory to save report

    Returns:
        Report dictionary
    """
    report = {}

    # Training loss analysis
    if "train/loss" in metrics:
        steps, values = zip(*metrics["train/loss"], strict=True)
        report["train_loss"] = {
            "initial": values[0],
            "final": values[-1],
            "min": min(values),
            "improvement": values[0] - values[-1],
        }

    # Validation loss analysis
    if "eval/loss" in metrics:
        steps, values = zip(*metrics["eval/loss"], strict=True)
        best_idx = values.index(min(values))
        report["eval_loss"] = {
            "initial": values[0],
            "final": values[-1],
            "best": min(values),
            "best_step": steps[best_idx],
            "best_epoch_approx": best_idx + 1,
        }

    # Overfitting check
    if "train/loss" in metrics and "eval/loss" in metrics:
        train_final = report["train_loss"]["final"]
        eval_final = report["eval_loss"]["final"]
        gap = eval_final - train_final
        report["overfitting"] = {
            "train_eval_gap": gap,
            "is_overfitting": gap > 0.5,  # Heuristic threshold
        }

    # Save report
    report_path = output_dir / "convergence_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Convergence report saved to {report_path}")

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze training convergence")
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to TensorBoard log directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/training_curves"),
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Loading TensorBoard events...")
    metrics = load_tensorboard_events(args.log_dir)

    logger.info(f"Available metrics: {list(metrics.keys())}")
    logger.info("")

    logger.info("Generating plots...")
    plot_training_curves(metrics, args.output_dir)

    logger.info("Generating convergence report...")
    report = generate_convergence_report(metrics, args.output_dir)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Convergence Summary")
    logger.info("=" * 60)
    if "train_loss" in report:
        tl = report["train_loss"]
        logger.info(
            f"  Train loss: {tl['initial']:.4f} -> {tl['final']:.4f} (improvement: {tl['improvement']:.4f})"
        )
    if "eval_loss" in report:
        el = report["eval_loss"]
        logger.info(
            f"  Eval loss: {el['initial']:.4f} -> {el['final']:.4f} (best: {el['best']:.4f} at step {el['best_step']})"
        )
    if "overfitting" in report:
        of = report["overfitting"]
        logger.info(
            f"  Train/eval gap: {of['train_eval_gap']:.4f} ({'OVERFITTING' if of['is_overfitting'] else 'OK'})"
        )


if __name__ == "__main__":
    main()
