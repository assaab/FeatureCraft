"""LLM provider abstraction for AI-powered feature engineering.

This module provides a provider-agnostic interface for LLM interactions,
supporting multiple backends (OpenAI, Anthropic, Azure OpenAI, local models).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Protocol

from ..logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Provider Protocol
# ============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Call LLM with prompt and return response."""
        ...
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost in USD for given token count."""
        ...


# ============================================================================
# Base Provider
# ============================================================================

class BaseLLMProvider(ABC):
    """Base class for LLM providers with common utilities."""
    
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize provider.
        
        Args:
            model: Model name/identifier
            api_key: API key (if None, read from env)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts on failure
        """
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
    
    @abstractmethod
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Call LLM with prompt."""
        pass
    
    @abstractmethod
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token count."""
        pass
    
    @staticmethod
    def hash_string(s: str) -> str:
        """Compute SHA256 hash of string."""
        return hashlib.sha256(s.encode()).hexdigest()


# ============================================================================
# OpenAI Provider
# ============================================================================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4, GPT-4o, GPT-5, etc.)."""
    
    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
            timeout: Request timeout
            max_retries: Max retries
        """
        super().__init__(model, api_key, timeout, max_retries)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
            )
    
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict[str, Any]:
        """Call OpenAI API with prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            tools: Function calling tools (optional)
            response_format: Response format spec (e.g., {"type": "json_object"})
            temperature: Sampling temperature (0.0 = deterministic)
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            Dict with keys: content, tokens_used, latency_ms, model
            
        Raises:
            ImportError: If openai package not installed
            Exception: On API errors after retries
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package required. Install with: pip install openai>=1.0.0"
            )
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request params
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
            **kwargs,
        }
        
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        if response_format:
            request_params["response_format"] = response_format
        
        # Call API with retries
        client = openai.OpenAI(api_key=self.api_key)
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = client.chat.completions.create(**request_params)
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Extract response
                message = response.choices[0].message
                content = message.content or ""
                
                # Handle tool calls if present
                tool_calls = []
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = [
                        {
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        }
                        for tc in message.tool_calls
                    ]
                
                # Token usage
                tokens_used = response.usage.total_tokens if response.usage else 0
                
                result = {
                    "content": content,
                    "tool_calls": tool_calls,
                    "tokens_used": tokens_used,
                    "latency_ms": latency_ms,
                    "model": self.model,
                    "finish_reason": response.choices[0].finish_reason,
                }
                
                logger.debug(
                    f"OpenAI call succeeded: {tokens_used} tokens, {latency_ms}ms"
                )
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"OpenAI call attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        raise Exception(f"OpenAI call failed after {self.max_retries} attempts: {last_error}")
    
    def estimate_cost(self, tokens: int, input_ratio: float = 0.7) -> float:
        """Estimate cost for token count.
        
        Args:
            tokens: Total token count
            input_ratio: Ratio of input tokens (default 0.7 = 70% input, 30% output)
            
        Returns:
            Estimated cost in USD
        """
        pricing = self.PRICING.get(self.model, {"input": 10.0, "output": 30.0})
        input_tokens = int(tokens * input_ratio)
        output_tokens = tokens - input_tokens
        
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )
        return cost


# ============================================================================
# Anthropic Provider
# ============================================================================

class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude models)."""
    
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    }
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize Anthropic provider."""
        super().__init__(model, api_key, timeout, max_retries)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var."
            )
    
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict[str, Any]:
        """Call Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package required. Install with: pip install anthropic>=0.18.0"
            )
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Build request
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        if tools:
            request_params["tools"] = tools
        
        # Call API with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = client.messages.create(**request_params)
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Extract content
                content = ""
                tool_calls = []
                
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "name": block.name,
                            "arguments": block.input,
                        })
                
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
                
                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "tokens_used": tokens_used,
                    "latency_ms": latency_ms,
                    "model": self.model,
                    "finish_reason": response.stop_reason,
                }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Anthropic call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise Exception(f"Anthropic call failed after {self.max_retries} attempts: {last_error}")
    
    def estimate_cost(self, tokens: int, input_ratio: float = 0.7) -> float:
        """Estimate cost for token count."""
        pricing = self.PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        input_tokens = int(tokens * input_ratio)
        output_tokens = tokens - input_tokens
        
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )
        return cost


# ============================================================================
# Mock Provider (for testing)
# ============================================================================

class MockLLMProvider(BaseLLMProvider):
    """Mock provider for testing without API calls."""
    
    def __init__(self, model: str = "mock", **kwargs):
        """Initialize mock provider."""
        super().__init__(model, **kwargs)
        self.call_count = 0
    
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Return mock response."""
        self.call_count += 1
        
        # Extract numeric column from prompt context if possible
        # Look for common column patterns mentioned in prompt
        numeric_col = "amount"  # default fallback
        
        # Try to find actual column names from prompt
        if "TransactionAmt" in prompt:
            numeric_col = "TransactionAmt"
        elif "Amount" in prompt:
            numeric_col = "Amount"
        elif "price" in prompt.lower():
            numeric_col = "price"
        elif "value" in prompt.lower():
            numeric_col = "value"
        
        # Mock JSON response for feature planning
        mock_content = json.dumps({
            "version": "1.0",
            "dataset_id": "mock_dataset",
            "task": "classification",
            "estimator_family": "tree",
            "candidates": [
                {
                    "name": "mock_feature_1",
                    "type": "rolling_mean",
                    "source_col": numeric_col,
                    "window": "30d",
                    "rationale": "Mock feature for testing",
                    "safety_tags": ["time_safe"],
                }
            ],
            "rationale": "Mock plan for testing",
        })
        
        return {
            "content": mock_content,
            "tool_calls": [],
            "tokens_used": 500,
            "latency_ms": 100,
            "model": "mock",
            "finish_reason": "stop",
        }
    
    def estimate_cost(self, tokens: int, input_ratio: float = 0.7) -> float:
        """Mock cost estimation."""
        return 0.0


# ============================================================================
# Provider Factory
# ============================================================================

def get_provider(
    provider_name: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """Get LLM provider by name.
    
    Args:
        provider_name: Provider name (openai, anthropic, mock)
        model: Model name (optional, uses provider default)
        api_key: API key (optional, uses env var)
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Initialized LLM provider
        
    Example:
        >>> provider = get_provider("openai", model="gpt-4o")
        >>> response = provider.call("What is 2+2?")
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mock": MockLLMProvider,
    }
    
    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {list(providers.keys())}"
        )
    
    provider_cls = providers[provider_name]
    
    # Set default models
    if model is None:
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "mock": "mock",
        }
        model = defaults.get(provider_name, "gpt-4o")
    
    return provider_cls(model=model, api_key=api_key, **kwargs)

