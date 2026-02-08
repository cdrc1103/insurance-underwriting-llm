"""Lightweight model provider abstraction using LiteLLM.

Provides a unified interface for calling any LLM provider (Anthropic, OpenAI,
Cohere, etc.) through LiteLLM. Includes retry logic, cost tracking, and
token usage reporting.

Example:
    provider = create_provider("claude-3-5-sonnet-20241022")
    response = provider.generate("What is 2+2?")
    print(response.content, response.cost_usd)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a model provider.

    Attributes:
        content: Generated text content
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        model_name: Model identifier used for generation
        cost_usd: Estimated cost in USD
        latency_ms: Time taken for the API call in milliseconds
    """

    content: str
    input_tokens: int
    output_tokens: int
    model_name: str
    cost_usd: float
    latency_ms: float


class LiteLLMProvider:
    """LiteLLM-based model provider supporting 100+ LLM providers.

    Uses LiteLLM's unified completion() interface to support any model
    provider. API keys are read from environment variables automatically
    (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY).

    Attributes:
        model: Model identifier (e.g., "claude-3-5-sonnet-20241022", "gpt-4o")
        temperature: Default sampling temperature
        max_tokens: Default maximum tokens to generate
        max_retries: Maximum number of retry attempts on failure
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        max_retries: int = 3,
    ) -> None:
        """Initialize provider.

        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022", "gpt-4o")
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens to generate
            max_retries: Maximum number of retry attempts on transient failures

        Raises:
            ValueError: If model identifier is empty
        """
        if not model:
            raise ValueError("Model identifier must not be empty")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: Input prompt text
            **kwargs: Override defaults (temperature, max_tokens)

        Returns:
            ModelResponse with content, token usage, cost, and latency

        Raises:
            APIError: If the API call fails after all retries
            APIConnectionError: If unable to connect to the API
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_api(prompt, temperature, max_tokens)
            except RateLimitError as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    "Rate limit hit (attempt %d/%d), waiting %ds",
                    attempt,
                    self.max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
            except (APIError, APIConnectionError) as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    "API error (attempt %d/%d): %s, retrying in %ds",
                    attempt,
                    self.max_retries,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)

        if last_error is None:
            raise RuntimeError("API call failed but no error was captured")
        raise last_error

    def _call_api(self, prompt: str, temperature: float, max_tokens: int) -> ModelResponse:
        """Make a single API call via LiteLLM.

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ModelResponse with content, token usage, cost, and latency
        """
        start_time = time.time()

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000

        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        content = response.choices[0].message.content
        if content is None:
            raise APIError(
                message="Model returned empty content", llm_provider=self.model, model=self.model
            )

        return ModelResponse(
            content=content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model_name=self.model,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def get_model_name(self) -> str:
        """Get model identifier.

        Returns:
            The model identifier string
        """
        return self.model


def create_provider(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    max_tokens: int = 100,
    max_retries: int = 3,
) -> LiteLLMProvider:
    """Factory function to create a model provider.

    Args:
        model: Model identifier (e.g., "claude-3-5-sonnet-20241022", "gpt-4o")
        temperature: Default sampling temperature
        max_tokens: Default maximum tokens to generate
        max_retries: Maximum retry attempts on transient failures

    Returns:
        Configured LiteLLMProvider instance

    Example:
        >>> provider = create_provider("claude-3-5-sonnet-20241022")
        >>> response = provider.generate("What is 2+2?")
        >>> print(response.content)

    Note:
        API keys are read from environment variables:
        - ANTHROPIC_API_KEY for Claude models
        - OPENAI_API_KEY for OpenAI models
    """
    return LiteLLMProvider(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
