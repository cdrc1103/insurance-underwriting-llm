"""Tests for model adapter export and loading."""

import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.export import (
    export_adapter,
    load_finetuned_model,
    merge_adapter,
)


@pytest.fixture(scope="module")
def peft_model_and_tokenizer():
    """Create a small PeftModel for testing."""
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)
    return peft_model, tokenizer


class TestExportAdapter:
    """Tests for export_adapter."""

    def test_saves_adapter_files(self, peft_model_and_tokenizer, tmp_path):
        """Test that adapter files are saved."""
        model, tokenizer = peft_model_and_tokenizer
        output = export_adapter(model, tokenizer, tmp_path / "adapter")

        assert output.exists()
        assert (output / "adapter_config.json").exists()
        # Check tokenizer files
        assert (output / "tokenizer_config.json").exists()

    def test_saves_metadata(self, peft_model_and_tokenizer, tmp_path):
        """Test that metadata is saved with git hash and timestamp."""
        model, tokenizer = peft_model_and_tokenizer
        output = export_adapter(
            model,
            tokenizer,
            tmp_path / "adapter",
            metrics={"loss": 0.5},
        )

        metadata_path = output / "training_metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "timestamp" in metadata
        assert "git_hash" in metadata
        assert "metrics" in metadata
        assert metadata["metrics"]["loss"] == 0.5

    def test_saves_training_config(self, peft_model_and_tokenizer, tmp_path):
        """Test that training config is saved."""
        model, tokenizer = peft_model_and_tokenizer
        config = {"lr": 2e-4, "epochs": 3}
        output = export_adapter(
            model,
            tokenizer,
            tmp_path / "adapter",
            training_config=config,
        )

        config_path = output / "training_args.json"
        assert config_path.exists()

        with open(config_path) as f:
            saved_config = json.load(f)
        assert saved_config["lr"] == 2e-4

    def test_non_peft_model_raises(self, tmp_path):
        """Test that non-PeftModel raises ValueError."""
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        with pytest.raises(ValueError, match="must be a PeftModel"):
            export_adapter(model, tokenizer, tmp_path)


class TestLoadFinetunedModel:
    """Tests for load_finetuned_model."""

    def test_load_roundtrip(self, peft_model_and_tokenizer, tmp_path):
        """Test save then load produces working model."""
        model, tokenizer = peft_model_and_tokenizer
        adapter_dir = tmp_path / "adapter"
        export_adapter(model, tokenizer, adapter_dir)

        loaded_model, loaded_tokenizer = load_finetuned_model(
            base_model_name="Qwen/Qwen3-0.6B",
            adapter_path=adapter_dir,
            device_map="cpu",
            dtype=torch.float32,
        )

        assert loaded_model is not None
        assert loaded_tokenizer is not None
        assert hasattr(loaded_model, "peft_config")

    def test_nonexistent_path_raises(self):
        """Test that nonexistent adapter path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            load_finetuned_model("Qwen/Qwen3-0.6B", "/nonexistent/path")


class TestMergeAdapter:
    """Tests for merge_adapter."""

    def test_merge_produces_base_model(self, peft_model_and_tokenizer):
        """Test that merging produces a non-PEFT model."""
        model, _ = peft_model_and_tokenizer
        merged = merge_adapter(model)

        # Should not be a PeftModel wrapper anymore
        from peft import PeftModel

        assert not isinstance(merged, PeftModel)

    def test_non_peft_model_raises(self):
        """Test that non-PeftModel raises ValueError."""
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        with pytest.raises(ValueError, match="must be a PeftModel"):
            merge_adapter(model)
