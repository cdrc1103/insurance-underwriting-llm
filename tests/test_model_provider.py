"""Tests for model provider abstraction layer."""

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.model_provider import (
    LiteLLMProvider,
    ModelResponse,
    create_provider,
)


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_creation(self):
        """Test basic creation with all fields."""
        response = ModelResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            model_name="test-model",
            cost_usd=0.001,
            latency_ms=100.5,
        )
        assert response.content == "Hello"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.model_name == "test-model"
        assert response.cost_usd == 0.001
        assert response.latency_ms == 100.5


class TestLiteLLMProvider:
    """Tests for LiteLLMProvider."""

    def test_init(self):
        """Test provider initialization with defaults."""
        provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.temperature == 0.0
        assert provider.max_tokens == 100
        assert provider.max_retries == 3

    def test_init_custom_params(self):
        """Test provider initialization with custom params."""
        provider = LiteLLMProvider(
            model="gpt-4o",
            temperature=1.5,
            max_tokens=200,
            max_retries=5,
        )
        assert provider.model == "gpt-4o"
        assert provider.temperature == 1.5
        assert provider.max_tokens == 200
        assert provider.max_retries == 5

    def test_init_empty_model_raises(self):
        """Test that empty model identifier raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            LiteLLMProvider(model="")

    def test_get_model_name(self):
        """Test get_model_name returns model identifier."""
        provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022")
        assert provider.get_model_name() == "claude-3-5-sonnet-20241022"

    @patch("src.evaluation.model_provider.litellm")
    def test_generate_calls_litellm(self, mock_litellm):
        """Test generate calls litellm.completion with correct params."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer: 4"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 3
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0005

        provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022", temperature=0.0)
        result = provider.generate("What is 2+2?")

        mock_litellm.completion.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.0,
            max_tokens=100,
        )
        assert result.content == "Answer: 4"
        assert result.input_tokens == 15
        assert result.output_tokens == 3
        assert result.cost_usd == 0.0005

    @patch("src.evaluation.model_provider.litellm")
    def test_generate_kwargs_override(self, mock_litellm):
        """Test generate allows overriding temperature and max_tokens."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 1
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.001

        provider = LiteLLMProvider(model="gpt-4o", temperature=0.0, max_tokens=100)
        provider.generate("test", temperature=2.0, max_tokens=50)

        mock_litellm.completion.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            temperature=2.0,
            max_tokens=50,
        )

    @patch("src.evaluation.model_provider.time.sleep")
    @patch("src.evaluation.model_provider.litellm")
    def test_generate_retries_on_rate_limit(self, mock_litellm, mock_sleep):
        """Test retry logic on rate limit errors."""
        from litellm.exceptions import RateLimitError

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 1
        mock_litellm.completion_cost.return_value = 0.001

        # First call fails, second succeeds
        mock_litellm.completion.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                llm_provider="anthropic",
                model="claude-3-5-sonnet-20241022",
            ),
            mock_response,
        ]

        provider = LiteLLMProvider(model="claude-3-5-sonnet-20241022", max_retries=3)
        result = provider.generate("test")

        assert result.content == "ok"
        assert mock_litellm.completion.call_count == 2
        mock_sleep.assert_called_once()


class TestCreateProvider:
    """Tests for create_provider factory function."""

    def test_default_model(self):
        """Test factory with default parameters."""
        provider = create_provider()
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.temperature == 0.0
        assert provider.max_tokens == 100

    def test_custom_model(self):
        """Test factory with custom model."""
        provider = create_provider(model="gpt-4o", temperature=1.0, max_tokens=200)
        assert provider.model == "gpt-4o"
        assert provider.temperature == 1.0
        assert provider.max_tokens == 200

    def test_returns_litellm_provider(self):
        """Test factory returns LiteLLMProvider instance."""
        provider = create_provider()
        assert isinstance(provider, LiteLLMProvider)
