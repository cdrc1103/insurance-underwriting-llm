"""Tests for inference pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.evaluation.inference import (
    EvaluationResult,
    GenerationConfig,
    GenerationResult,
    batch_generate_responses,
    evaluate_dataset,
    evaluate_dataset_batched,
    generate_response,
    generate_response_with_metadata,
    save_evaluation_results,
)


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.1
        assert config.stop_strings == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
            stop_strings=["STOP", "END"],
        )

        assert config.max_new_tokens == 256
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.repetition_penalty == 1.2
        assert config.stop_strings == ["STOP", "END"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = GenerationConfig(max_new_tokens=100, temperature=0.3)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["max_new_tokens"] == 100
        assert config_dict["temperature"] == 0.3
        assert "stop_strings" in config_dict

    def test_validation_max_new_tokens(self):
        """Test validation of max_new_tokens parameter."""
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            GenerationConfig(max_new_tokens=0)

    def test_validation_temperature(self):
        """Test validation of temperature parameter."""
        with pytest.raises(ValueError, match="temperature must be >= 0.0"):
            GenerationConfig(temperature=-0.1)

    def test_validation_top_p(self):
        """Test validation of top_p parameter."""
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=-0.1)

    def test_validation_top_k(self):
        """Test validation of top_k parameter."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            GenerationConfig(top_k=0)

    def test_validation_repetition_penalty(self):
        """Test validation of repetition_penalty parameter."""
        with pytest.raises(ValueError, match="repetition_penalty must be >= 1.0"):
            GenerationConfig(repetition_penalty=0.5)


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_creation(self):
        """Test GenerationResult creation."""
        result = GenerationResult(
            response="Test response",
            generation_time_ms=150.5,
            input_tokens=50,
            output_tokens=20,
            config={"temperature": 0.7},
        )

        assert result.response == "Test response"
        assert result.generation_time_ms == 150.5
        assert result.input_tokens == 50
        assert result.output_tokens == 20
        assert result.config == {"temperature": 0.7}


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Test EvaluationResult creation."""
        result = EvaluationResult(
            results=[{"response": "test"}],
            config={"temperature": 0.7},
            total_time_ms=1000.0,
            successful_count=1,
            failed_count=0,
        )

        assert len(result.results) == 1
        assert result.total_time_ms == 1000.0
        assert result.successful_count == 1
        assert result.failed_count == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            results=[{"response": "test"}],
            config={"temperature": 0.7},
            total_time_ms=1000.0,
            successful_count=1,
            failed_count=0,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "results" in result_dict
        assert "config" in result_dict
        assert "total_time_ms" in result_dict


class TestGenerateResponse:
    """Tests for generate_response function."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<|im_start|>user\nHello<|im_end|>")
        tokenizer.pad_token_id = 0
        tokenizer.decode = MagicMock(return_value="Generated response")

        # Mock tokenizer call returns dict-like object
        mock_inputs = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        tokenizer.return_value = MagicMock()
        tokenizer.return_value.to = MagicMock(return_value=mock_inputs)
        tokenizer.return_value.__getitem__ = lambda _self, key: mock_inputs[key]

        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate = MagicMock(return_value=torch.zeros(1, 20, dtype=torch.long))

        return model, tokenizer

    def test_basic_generation(self, mock_model_and_tokenizer):
        """Test basic response generation."""
        model, tokenizer = mock_model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]

        response = generate_response(model, tokenizer, messages)

        assert isinstance(response, str)
        model.generate.assert_called_once()
        tokenizer.apply_chat_template.assert_called_once()

    def test_generation_with_config(self, mock_model_and_tokenizer):
        """Test generation with GenerationConfig."""
        model, tokenizer = mock_model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_new_tokens=100, temperature=0.5)

        response = generate_response(model, tokenizer, messages, config=config)

        assert isinstance(response, str)
        # Verify generate was called with correct params
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5

    def test_generation_with_stop_strings(self, mock_model_and_tokenizer):
        """Test generation with stop strings."""
        model, tokenizer = mock_model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(stop_strings=["STOP"])

        response = generate_response(model, tokenizer, messages, config=config)

        assert isinstance(response, str)
        # Verify stopping criteria was passed
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["stopping_criteria"] is not None

    def test_greedy_decoding_with_zero_temperature(self, mock_model_and_tokenizer):
        """Test greedy decoding when temperature is 0."""
        model, tokenizer = mock_model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(temperature=0.0)

        generate_response(model, tokenizer, messages, config=config)

        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False
        assert call_kwargs.get("temperature") is None


class TestGenerateResponseWithMetadata:
    """Tests for generate_response_with_metadata function."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        tokenizer.pad_token_id = 0
        tokenizer.decode = MagicMock(return_value="Generated response")

        mock_inputs = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        tokenizer.return_value = MagicMock()
        tokenizer.return_value.to = MagicMock(return_value=mock_inputs)
        tokenizer.return_value.__getitem__ = lambda _self, key: mock_inputs[key]

        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate = MagicMock(return_value=torch.zeros(1, 20, dtype=torch.long))

        return model, tokenizer

    def test_returns_generation_result(self, mock_model_and_tokenizer):
        """Test that function returns GenerationResult."""
        model, tokenizer = mock_model_and_tokenizer
        messages = [{"role": "user", "content": "Hello"}]

        result = generate_response_with_metadata(model, tokenizer, messages)

        assert isinstance(result, GenerationResult)
        assert result.response is not None
        assert result.generation_time_ms >= 0
        assert result.input_tokens > 0
        assert result.output_tokens >= 0
        assert isinstance(result.config, dict)


class TestEvaluateDataset:
    """Tests for evaluate_dataset function."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        return [
            {
                "messages": [{"role": "user", "content": "Question 1"}],
                "task": "task1",
                "target_response": "Answer 1",
                "original_index": 0,
            },
            {
                "messages": [{"role": "user", "content": "Question 2"}],
                "task": "task2",
                "target_response": "Answer 2",
                "original_index": 1,
            },
        ]

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        tokenizer.pad_token_id = 0
        tokenizer.decode = MagicMock(return_value="Generated response")

        mock_inputs = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        tokenizer.return_value = MagicMock()
        tokenizer.return_value.to = MagicMock(return_value=mock_inputs)
        tokenizer.return_value.__getitem__ = lambda _self, key: mock_inputs[key]

        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate = MagicMock(return_value=torch.zeros(1, 20, dtype=torch.long))

        return model, tokenizer

    def test_evaluates_all_examples(self, mock_model_and_tokenizer, mock_dataset):
        """Test that all examples are evaluated."""
        model, tokenizer = mock_model_and_tokenizer

        result = evaluate_dataset(model, tokenizer, mock_dataset)

        assert isinstance(result, EvaluationResult)
        assert len(result.results) == 2
        assert result.successful_count == 2
        assert result.failed_count == 0

    def test_includes_metadata_in_results(self, mock_model_and_tokenizer, mock_dataset):
        """Test that results include all expected metadata."""
        model, tokenizer = mock_model_and_tokenizer

        result = evaluate_dataset(model, tokenizer, mock_dataset)

        for r in result.results:
            assert "original_index" in r
            assert "task" in r
            assert "messages" in r
            assert "target_response" in r
            assert "generated_response" in r
            assert "generation_time_ms" in r
            assert "memory_used_mb" in r
            assert "memory_delta_mb" in r
            assert "input_tokens" in r
            assert "output_tokens" in r

    def test_raises_on_empty_dataset(self, mock_model_and_tokenizer):
        """Test that empty dataset raises ValueError."""
        model, tokenizer = mock_model_and_tokenizer

        with pytest.raises(ValueError, match="Dataset is empty"):
            evaluate_dataset(model, tokenizer, [])

    def test_raises_on_missing_messages_field(self, mock_model_and_tokenizer):
        """Test that missing messages field raises ValueError."""
        model, tokenizer = mock_model_and_tokenizer
        invalid_dataset = [{"task": "task1"}]

        with pytest.raises(ValueError, match="missing required 'messages' field"):
            evaluate_dataset(model, tokenizer, invalid_dataset)


class TestBatchGenerateResponses:
    """Tests for batch_generate_responses function."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        tokenizer.pad_token_id = 0
        tokenizer.decode = MagicMock(return_value="Generated response")
        tokenizer.batch_decode = MagicMock(
            return_value=["Generated response 1", "Generated response 2"]
        )

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda _self, _key: torch.zeros(2, 10, dtype=torch.long)
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        tokenizer.return_value = mock_inputs

        model = MagicMock()
        model.device = torch.device("cpu")
        # Return batch of outputs
        model.generate = MagicMock(return_value=torch.zeros(2, 20, dtype=torch.long))

        return model, tokenizer

    def test_batch_generation(self, mock_model_and_tokenizer):
        """Test batch generation returns correct number of results."""
        model, tokenizer = mock_model_and_tokenizer
        messages_batch = [
            [{"role": "user", "content": "Q1"}],
            [{"role": "user", "content": "Q2"}],
        ]

        results = batch_generate_responses(model, tokenizer, messages_batch)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, GenerationResult)

    def test_raises_on_empty_batch(self, mock_model_and_tokenizer):
        """Test that empty batch raises ValueError."""
        model, tokenizer = mock_model_and_tokenizer

        with pytest.raises(ValueError, match="Batch is empty"):
            batch_generate_responses(model, tokenizer, [])


class TestEvaluateDatasetBatched:
    """Tests for evaluate_dataset_batched function."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        return [
            {
                "messages": [{"role": "user", "content": "Question 1"}],
                "task": "task1",
                "target_response": "Answer 1",
                "original_index": 0,
            },
            {
                "messages": [{"role": "user", "content": "Question 2"}],
                "task": "task2",
                "target_response": "Answer 2",
                "original_index": 1,
            },
        ]

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        tokenizer.pad_token_id = 0
        tokenizer.decode = MagicMock(return_value="Generated response")

        mock_inputs = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        tokenizer.return_value = MagicMock()
        tokenizer.return_value.to = MagicMock(return_value=mock_inputs)
        tokenizer.return_value.__getitem__ = lambda _self, key: mock_inputs[key]

        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate = MagicMock(return_value=torch.zeros(1, 20, dtype=torch.long))

        return model, tokenizer

    def test_fallback_to_sequential_on_batch_error(
        self, mock_model_and_tokenizer, mock_dataset, caplog
    ):
        """Test that batch processing falls back to sequential on error with logging."""
        import logging

        model, tokenizer = mock_model_and_tokenizer

        # Make batch generation fail but keep sequential generation working
        from unittest.mock import patch

        with patch("src.evaluation.inference.batch_generate_responses") as mock_batch_gen:
            # Make batch_generate_responses raise an exception
            mock_batch_gen.side_effect = RuntimeError("Batch processing failed")

            # Enable logging capture
            with caplog.at_level(logging.WARNING):
                result = evaluate_dataset_batched(model, tokenizer, mock_dataset, batch_size=2)

            # Verify fallback succeeded
            assert isinstance(result, EvaluationResult)
            assert result.successful_count == 2
            assert result.failed_count == 0
            assert len(result.results) == 2

            # Verify warning was logged
            assert any("Batch processing failed" in record.message for record in caplog.records)
            assert any("Falling back to sequential" in record.message for record in caplog.records)


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""

    def test_saves_to_json(self):
        """Test saving results to JSON file."""
        result = EvaluationResult(
            results=[{"response": "test", "messages": [{"role": "user", "content": "hi"}]}],
            config={"temperature": 0.7},
            total_time_ms=1000.0,
            successful_count=1,
            failed_count=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            save_evaluation_results(result, output_path)

            assert output_path.exists()
            with open(output_path) as f:
                saved_data = json.load(f)

            assert "results" in saved_data
            assert "config" in saved_data
            assert saved_data["successful_count"] == 1

    def test_excludes_messages_when_requested(self):
        """Test excluding messages from saved results."""
        result = EvaluationResult(
            results=[{"response": "test", "messages": [{"role": "user", "content": "hi"}]}],
            config={"temperature": 0.7},
            total_time_ms=1000.0,
            successful_count=1,
            failed_count=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            save_evaluation_results(result, output_path, include_messages=False)

            with open(output_path) as f:
                saved_data = json.load(f)

            assert "messages" not in saved_data["results"][0]

    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        result = EvaluationResult(
            results=[],
            config={},
            total_time_ms=0,
            successful_count=0,
            failed_count=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "results.json"
            save_evaluation_results(result, output_path)

            assert output_path.exists()


class TestIntegration:
    """Integration tests requiring actual model loading."""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load actual model for integration tests."""
        try:
            from src.models.model_loader import load_base_model

            model, tokenizer = load_base_model(
                model_name="Qwen/Qwen3-0.6B",
                device_map="cpu",
                dtype=torch.float32,
            )
            return model, tokenizer
        except Exception:
            pytest.skip("Model loading failed, skipping integration tests")

    def test_generate_response_integration(self, loaded_model):
        """Test actual response generation."""
        model, tokenizer = loaded_model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        config = GenerationConfig(max_new_tokens=20, temperature=0.0)
        response = generate_response(model, tokenizer, messages, config=config)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_evaluate_dataset_integration(self, loaded_model):
        """Test dataset evaluation with actual model."""
        model, tokenizer = loaded_model
        dataset = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hello."},
                ],
                "task": "greeting",
                "target_response": "Hello!",
            }
        ]

        config = GenerationConfig(max_new_tokens=10, temperature=0.0)
        result = evaluate_dataset(model, tokenizer, dataset, config=config)

        assert result.successful_count == 1
        assert result.failed_count == 0
        assert len(result.results) == 1
        assert result.results[0]["generated_response"] is not None
