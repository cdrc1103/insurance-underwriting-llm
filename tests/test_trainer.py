"""Tests for training setup and configuration."""

import random

import numpy as np
import torch
from transformers import TrainingArguments

from src.training.trainer import (
    compute_metrics,
    create_training_args,
    set_seed,
)


class TestSetSeed:
    """Tests for set_seed."""

    def test_reproducibility(self):
        """Test that setting the same seed produces same random numbers."""
        set_seed(42)
        val1 = random.random()
        np_val1 = np.random.random()
        torch_val1 = torch.rand(1).item()

        set_seed(42)
        val2 = random.random()
        np_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()

        assert val1 == val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2

    def test_different_seeds_differ(self):
        """Test that different seeds produce different numbers."""
        set_seed(42)
        val1 = random.random()

        set_seed(123)
        val2 = random.random()

        assert val1 != val2


class TestCreateTrainingArgs:
    """Tests for create_training_args."""

    def test_returns_training_arguments(self, tmp_path):
        """Test that output is TrainingArguments."""
        args = create_training_args(output_dir=tmp_path, fp16=False)
        assert isinstance(args, TrainingArguments)

    def test_default_values(self, tmp_path):
        """Test that defaults match config constants."""
        args = create_training_args(output_dir=tmp_path, fp16=False)
        assert args.learning_rate == 2e-4
        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 2
        assert args.num_train_epochs == 3
        assert args.max_grad_norm == 1.0
        assert args.weight_decay == 0.01

    def test_custom_values(self, tmp_path):
        """Test that custom values override defaults."""
        args = create_training_args(
            output_dir=tmp_path,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            num_train_epochs=5,
            fp16=False,
        )
        assert args.learning_rate == 1e-4
        assert args.per_device_train_batch_size == 8
        assert args.num_train_epochs == 5

    def test_eval_strategy_configured(self, tmp_path):
        """Test that evaluation strategy is set."""
        args = create_training_args(output_dir=tmp_path, fp16=False)
        assert args.eval_strategy == "steps"

    def test_save_best_model(self, tmp_path):
        """Test that load_best_model_at_end is enabled."""
        args = create_training_args(output_dir=tmp_path, fp16=False)
        assert args.load_best_model_at_end is True
        assert args.metric_for_best_model == "eval_loss"
        assert args.greater_is_better is False

    def test_tensorboard_logging(self, tmp_path):
        """Test that TensorBoard reporting is configured."""
        args = create_training_args(output_dir=tmp_path, fp16=False)
        assert "tensorboard" in args.report_to


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_returns_perplexity(self):
        """Test that perplexity is computed from predictions."""
        # Create mock eval predictions
        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        labels = np.random.randint(0, vocab_size, (batch_size, seq_len))

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

        result = compute_metrics(MockEvalPred(logits, labels))
        assert "perplexity" in result
        assert result["perplexity"] > 0

    def test_masked_labels_excluded(self):
        """Test that -100 labels are excluded from perplexity."""
        batch_size, seq_len, vocab_size = 1, 10, 50
        logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        # All labels are -100 except last 3
        labels = np.full((batch_size, seq_len), -100)
        labels[0, -3:] = np.random.randint(0, vocab_size, 3)

        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

        result = compute_metrics(MockEvalPred(logits, labels))
        assert "perplexity" in result
        assert result["perplexity"] > 0
