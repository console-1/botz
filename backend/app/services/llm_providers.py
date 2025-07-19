"""
LLM Provider Service

This module provides abstraction for multiple LLM providers including:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Mistral AI
- Venice.ai
- OpenRouter (unified API for multiple models)
- Ollama (self-hosted models)

Supports fallback chains, cost optimization, and provider-specific configurations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import json
import logging
import httpx
from pydantic import BaseModel, Field

# LLM provider imports
import openai
import anthropic
import ollama

from ..core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    VENICE_AI = "venice_ai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class LLMModelType(str, Enum):
    """Categories of LLM models"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"


@dataclass
class LLMModel:
    """Model configuration for LLM providers"""
    provider: LLMProvider
    model_name: str
    model_type: LLMModelType
    max_tokens: int
    cost_per_1k_tokens: float
    supports_streaming: bool = True
    supports_function_calling: bool = False
    context_window: int = 4096
    
    def __str__(self):
        return f"{self.provider.value}/{self.model_name}"


@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int
    cost_usd: float
    processing_time_ms: int
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMRequest:
    """Standardized request to LLM providers"""
    messages: List[Dict[str, str]]
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Model configurations for different providers
LLM_MODELS = {
    LLMProvider.OPENAI: {
        "gpt-4": LLMModel(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            model_type=LLMModelType.CHAT,
            max_tokens=8192,
            cost_per_1k_tokens=0.03,
            supports_function_calling=True,
            context_window=8192
        ),
        "gpt-4-turbo": LLMModel(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4-turbo",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.01,
            supports_function_calling=True,
            context_window=128000
        ),
        "gpt-3.5-turbo": LLMModel(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.001,
            supports_function_calling=True,
            context_window=16385
        ),
    },
    LLMProvider.ANTHROPIC: {
        "claude-3-5-sonnet": LLMModel(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.003,
            supports_function_calling=True,
            context_window=200000
        ),
        "claude-3-5-haiku": LLMModel(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-5-haiku-20241022",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.00025,
            supports_function_calling=True,
            context_window=200000
        ),
        "claude-3-opus": LLMModel(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.015,
            supports_function_calling=True,
            context_window=200000
        ),
    },
    LLMProvider.MISTRAL: {
        "mistral-large": LLMModel(
            provider=LLMProvider.MISTRAL,
            model_name="mistral-large-latest",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.008,
            supports_function_calling=True,
            context_window=32768
        ),
        "mistral-small": LLMModel(
            provider=LLMProvider.MISTRAL,
            model_name="mistral-small-latest",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.002,
            supports_function_calling=True,
            context_window=32768
        ),
        "mixtral-8x7b": LLMModel(
            provider=LLMProvider.MISTRAL,
            model_name="open-mixtral-8x7b",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.0007,
            supports_function_calling=False,
            context_window=32768
        ),
    },
    LLMProvider.VENICE_AI: {
        "llama-3-8b": LLMModel(
            provider=LLMProvider.VENICE_AI,
            model_name="llama-3-8b-instruct",
            model_type=LLMModelType.CHAT,
            max_tokens=2048,
            cost_per_1k_tokens=0.0003,
            supports_function_calling=False,
            context_window=8192
        ),
        "llama-3-70b": LLMModel(
            provider=LLMProvider.VENICE_AI,
            model_name="llama-3-70b-instruct",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.0009,
            supports_function_calling=False,
            context_window=8192
        ),
    },
    LLMProvider.OPENROUTER: {
        "gpt-4": LLMModel(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4",
            model_type=LLMModelType.CHAT,
            max_tokens=8192,
            cost_per_1k_tokens=0.03,
            supports_function_calling=True,
            context_window=8192
        ),
        "claude-3-5-sonnet": LLMModel(
            provider=LLMProvider.OPENROUTER,
            model_name="anthropic/claude-3-5-sonnet",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.003,
            supports_function_calling=True,
            context_window=200000
        ),
        "llama-3-8b": LLMModel(
            provider=LLMProvider.OPENROUTER,
            model_name="meta-llama/llama-3-8b-instruct",
            model_type=LLMModelType.CHAT,
            max_tokens=2048,
            cost_per_1k_tokens=0.0001,
            supports_function_calling=False,
            context_window=8192
        ),
    },
    LLMProvider.OLLAMA: {
        "llama3": LLMModel(
            provider=LLMProvider.OLLAMA,
            model_name="llama3",
            model_type=LLMModelType.CHAT,
            max_tokens=2048,
            cost_per_1k_tokens=0.0,  # Free for self-hosted
            supports_function_calling=False,
            context_window=8192
        ),
        "mistral": LLMModel(
            provider=LLMProvider.OLLAMA,
            model_name="mistral",
            model_type=LLMModelType.CHAT,
            max_tokens=2048,
            cost_per_1k_tokens=0.0,  # Free for self-hosted
            supports_function_calling=False,
            context_window=8192
        ),
        "codellama": LLMModel(
            provider=LLMProvider.OLLAMA,
            model_name="codellama",
            model_type=LLMModelType.CHAT,
            max_tokens=2048,
            cost_per_1k_tokens=0.0,  # Free for self-hosted
            supports_function_calling=False,
            context_window=16384
        ),
    },
}


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, provider: LLMProvider, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.models = LLM_MODELS.get(provider, {})
    
    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def stream_response(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible"""
        pass
    
    def get_model_info(self, model_name: str) -> Optional[LLMModel]:
        """Get model information"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[LLMModel]:
        """List available models for this provider"""
        return list(self.models.values())
    
    def calculate_cost(self, tokens: int, model_name: str) -> float:
        """Calculate cost based on token usage"""
        model = self.get_model_info(model_name)
        if not model:
            return 0.0
        return (tokens / 1000) * model.cost_per_1k_tokens


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(LLMProvider.OPENAI, api_key)
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API"""
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=request.stream,
                functions=request.functions
            )
            
            end_time = datetime.now(timezone.utc)
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            cost = self.calculate_cost(tokens_used, request.model)
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=self.provider,
                tokens_used=tokens_used,
                cost_usd=cost,
                processing_time_ms=processing_time_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def stream_response(self, request: LLMRequest):
        """Stream response using OpenAI API"""
        try:
            request.stream = True
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stream=True,
                functions=request.functions
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(LLMProvider.ANTHROPIC, api_key)
        self.client = anthropic.AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)
            
            response = await self.client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                system=system_message,
                messages=messages
            )
            
            end_time = datetime.now(timezone.utc)
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = self.calculate_cost(tokens_used, request.model)
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=self.provider,
                tokens_used=tokens_used,
                cost_usd=cost,
                processing_time_ms=processing_time_ms,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def stream_response(self, request: LLMRequest):
        """Stream response using Anthropic API"""
        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)
            
            async with self.client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                system=system_message,
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Anthropic API health"""
        try:
            response = await self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama self-hosted provider"""
    
    def __init__(self, base_url: Optional[str] = None):
        super().__init__(LLMProvider.OLLAMA, base_url=base_url)
        self.client = ollama.AsyncClient(host=base_url or settings.ollama_base_url)
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama API"""
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await self.client.chat(
                model=request.model,
                messages=request.messages,
                options={
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens
                }
            )
            
            end_time = datetime.now(timezone.utc)
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            content = response["message"]["content"]
            tokens_used = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
            cost = self.calculate_cost(tokens_used, request.model)  # Should be 0 for self-hosted
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=self.provider,
                tokens_used=tokens_used,
                cost_usd=cost,
                processing_time_ms=processing_time_ms,
                metadata={
                    "eval_count": response.get("eval_count", 0),
                    "prompt_eval_count": response.get("prompt_eval_count", 0),
                    "eval_duration": response.get("eval_duration", 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def stream_response(self, request: LLMRequest):
        """Stream response using Ollama API"""
        try:
            async for part in await self.client.chat(
                model=request.model,
                messages=request.messages,
                stream=True,
                options={
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens
                }
            ):
                if part["message"]["content"]:
                    yield part["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            models = await self.client.list()
            return len(models["models"]) > 0
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


class GenericAPIProvider(BaseLLMProvider):
    """Generic API provider for Venice.ai, OpenRouter, and Mistral"""
    
    def __init__(self, provider: LLMProvider, api_key: str, base_url: str):
        super().__init__(provider, api_key, base_url)
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using generic OpenAI-compatible API"""
        start_time = datetime.now(timezone.utc)
        
        try:
            payload = {
                "model": request.model,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": False
            }
            
            if request.stop:
                payload["stop"] = request.stop
            
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            end_time = datetime.now(timezone.utc)
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]
            cost = self.calculate_cost(tokens_used, request.model)
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=self.provider,
                tokens_used=tokens_used,
                cost_usd=cost,
                processing_time_ms=processing_time_ms,
                metadata={
                    "finish_reason": data["choices"][0]["finish_reason"],
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"]
                }
            )
            
        except Exception as e:
            logger.error(f"{self.provider.value} API error: {e}")
            raise
    
    async def stream_response(self, request: LLMRequest):
        """Stream response using generic API"""
        try:
            payload = {
                "model": request.model,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": True
            }
            
            if request.stop:
                payload["stop"] = request.stop
            
            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data != "[DONE]":
                            try:
                                chunk = json.loads(data)
                                if chunk["choices"][0]["delta"].get("content"):
                                    yield chunk["choices"][0]["delta"]["content"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"{self.provider.value} streaming error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check generic API health"""
        try:
            response = await self.client.get("/models")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"{self.provider.value} health check failed: {e}")
            return False


# Provider factory
def create_provider(provider: LLMProvider, **kwargs) -> BaseLLMProvider:
    """Factory function to create LLM providers"""
    if provider == LLMProvider.OPENAI:
        return OpenAIProvider(kwargs.get("api_key"))
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicProvider(kwargs.get("api_key"))
    elif provider == LLMProvider.OLLAMA:
        return OllamaProvider(kwargs.get("base_url"))
    elif provider == LLMProvider.MISTRAL:
        return GenericAPIProvider(
            provider,
            kwargs.get("api_key", settings.mistral_api_key),
            "https://api.mistral.ai/v1"
        )
    elif provider == LLMProvider.VENICE_AI:
        return GenericAPIProvider(
            provider,
            kwargs.get("api_key", settings.venice_ai_api_key),
            settings.venice_ai_base_url
        )
    elif provider == LLMProvider.OPENROUTER:
        return GenericAPIProvider(
            provider,
            kwargs.get("api_key", settings.openrouter_api_key),
            settings.openrouter_base_url
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")