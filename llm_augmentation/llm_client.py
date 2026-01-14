"""
LLM Client Abstraction for Career-Aware Data Augmentation

This module provides an abstract interface for LLM providers and concrete
implementations for supported providers (OpenAI, Anthropic).
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import LLMProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM generation request."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The input prompt for generation
            temperature: Optional override for temperature setting
            max_tokens: Optional override for max tokens setting
            
        Returns:
            LLMResponse containing the generated text and metadata
            
        Raises:
            LLMClientError: If generation fails after retries
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM client is properly configured and available."""
        pass


class LLMClientError(Exception):
    """Exception raised when LLM client operations fail."""
    pass


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    def __init__(
        self,
        config: LLMProviderConfig,
        max_retries: int = 3,
        retry_delay_base: float = 1.0,
        retry_delay_multiplier: float = 2.0
    ):
        """
        Initialize OpenAI client.
        
        Args:
            config: LLM provider configuration
            max_retries: Maximum number of retry attempts
            retry_delay_base: Base delay between retries in seconds
            retry_delay_multiplier: Multiplier for exponential backoff
        """
        self.config = config
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.retry_delay_multiplier = retry_delay_multiplier
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            logger.warning(
                f"API key not found in environment variable: {self.config.api_key_env}"
            )
            return
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {self.config.model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def is_available(self) -> bool:
        """Check if the OpenAI client is properly configured."""
        return self._client is not None
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: The input prompt for generation
            temperature: Optional override for temperature setting
            max_tokens: Optional override for max tokens setting
            
        Returns:
            LLMResponse containing the generated text and metadata
            
        Raises:
            LLMClientError: If generation fails after retries
        """
        if not self.is_available():
            raise LLMClientError("OpenAI client not available. Check API key configuration.")
        
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=tokens
                )
                
                return LLMResponse(
                    text=response.choices[0].message.content.strip(),
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    finish_reason=response.choices[0].finish_reason
                )
                
            except Exception as e:
                last_error = e
                delay = self.retry_delay_base * (self.retry_delay_multiplier ** attempt)
                logger.warning(
                    f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
        
        raise LLMClientError(f"Failed after {self.max_retries} attempts: {last_error}")


class AnthropicClient(LLMClient):
    """Anthropic API client implementation."""
    
    def __init__(
        self,
        config: LLMProviderConfig,
        max_retries: int = 3,
        retry_delay_base: float = 1.0,
        retry_delay_multiplier: float = 2.0
    ):
        """
        Initialize Anthropic client.
        
        Args:
            config: LLM provider configuration
            max_retries: Maximum number of retry attempts
            retry_delay_base: Base delay between retries in seconds
            retry_delay_multiplier: Multiplier for exponential backoff
        """
        self.config = config
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.retry_delay_multiplier = retry_delay_multiplier
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            logger.warning(
                f"API key not found in environment variable: {self.config.api_key_env}"
            )
            return
        
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
            logger.info(f"Anthropic client initialized with model: {self.config.model_name}")
        except ImportError:
            logger.error("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def is_available(self) -> bool:
        """Check if the Anthropic client is properly configured."""
        return self._client is not None
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: The input prompt for generation
            temperature: Optional override for temperature setting
            max_tokens: Optional override for max tokens setting
            
        Returns:
            LLMResponse containing the generated text and metadata
            
        Raises:
            LLMClientError: If generation fails after retries
        """
        if not self.is_available():
            raise LLMClientError("Anthropic client not available. Check API key configuration.")
        
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.config.model_name,
                    max_tokens=tokens,
                    temperature=temp,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return LLMResponse(
                    text=response.content[0].text.strip(),
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    },
                    finish_reason=response.stop_reason
                )
                
            except Exception as e:
                last_error = e
                delay = self.retry_delay_base * (self.retry_delay_multiplier ** attempt)
                logger.warning(
                    f"Anthropic API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
        
        raise LLMClientError(f"Failed after {self.max_retries} attempts: {last_error}")


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""
    
    def __init__(self, config: LLMProviderConfig):
        """Initialize mock client."""
        self.config = config
        self._responses: List[str] = []
        self._response_index = 0
        logger.info("Mock LLM client initialized")
    
    def set_responses(self, responses: List[str]) -> None:
        """Set predefined responses for testing."""
        self._responses = responses
        self._response_index = 0
    
    def is_available(self) -> bool:
        """Mock client is always available."""
        return True
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Return mock response.
        
        If predefined responses are set, returns them in order.
        Otherwise, returns a simple transformation of the prompt.
        """
        if self._responses:
            text = self._responses[self._response_index % len(self._responses)]
            self._response_index += 1
        else:
            # Simple mock transformation
            text = f"[Transformed] {prompt[:100]}..."
        
        return LLMResponse(
            text=text,
            model="mock-model",
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4, "total_tokens": (len(prompt) + len(text)) // 4},
            finish_reason="stop"
        )


def create_llm_client(
    config: LLMProviderConfig,
    max_retries: int = 3,
    retry_delay_base: float = 1.0,
    retry_delay_multiplier: float = 2.0
) -> LLMClient:
    """
    Factory function to create the appropriate LLM client based on configuration.
    
    Args:
        config: LLM provider configuration
        max_retries: Maximum number of retry attempts
        retry_delay_base: Base delay between retries in seconds
        retry_delay_multiplier: Multiplier for exponential backoff
        
    Returns:
        Configured LLM client instance
        
    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = config.provider_type.lower()
    
    if provider_type == "openai":
        return OpenAIClient(
            config=config,
            max_retries=max_retries,
            retry_delay_base=retry_delay_base,
            retry_delay_multiplier=retry_delay_multiplier
        )
    elif provider_type == "anthropic":
        return AnthropicClient(
            config=config,
            max_retries=max_retries,
            retry_delay_base=retry_delay_base,
            retry_delay_multiplier=retry_delay_multiplier
        )
    elif provider_type == "mock":
        return MockLLMClient(config=config)
    else:
        raise ValueError(f"Unsupported LLM provider type: {provider_type}")
