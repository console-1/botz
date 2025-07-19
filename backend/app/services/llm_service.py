"""
LLM Service - Abstraction Layer

This service provides a unified interface for interacting with multiple LLM providers.
Handles provider selection, fallback chains, error handling, and response optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timezone

from .llm_providers import (
    BaseLLMProvider, LLMProvider, LLMRequest, LLMResponse, 
    create_provider, LLM_MODELS
)
from ..core.config import settings

logger = logging.getLogger(__name__)


class FallbackStrategy(str, Enum):
    """Fallback strategies for LLM provider failures"""
    NONE = "none"  # No fallback, fail immediately
    NEXT_PROVIDER = "next_provider"  # Try next provider in chain
    CHEAPER_MODEL = "cheaper_model"  # Try cheaper model in same provider
    FASTEST_MODEL = "fastest_model"  # Try fastest available model
    BEST_AVAILABLE = "best_available"  # Try best available model


@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    primary_provider: LLMProvider = LLMProvider.OPENAI
    primary_model: str = "gpt-3.5-turbo"
    fallback_strategy: FallbackStrategy = FallbackStrategy.NEXT_PROVIDER
    fallback_providers: List[LLMProvider] = field(default_factory=list)
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: float = 30.0
    enable_streaming: bool = True
    enable_caching: bool = True
    cost_limit_per_request: float = 0.10  # USD
    
    def __post_init__(self):
        if not self.fallback_providers:
            # Default fallback chain
            self.fallback_providers = [
                LLMProvider.OPENAI,
                LLMProvider.ANTHROPIC,
                LLMProvider.MISTRAL,
                LLMProvider.VENICE_AI,
                LLMProvider.OPENROUTER,
                LLMProvider.OLLAMA
            ]


@dataclass
class LLMMetrics:
    """Metrics for LLM service monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    fallback_uses: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    provider_success_rates: Dict[LLMProvider, float] = field(default_factory=dict)
    
    def add_request(self, provider: LLMProvider, success: bool, tokens: int, cost: float, response_time: float):
        """Add request metrics"""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time) / self.total_requests
        )
        
        # Update provider success rates
        if provider not in self.provider_success_rates:
            self.provider_success_rates[provider] = 0.0
        
        # Simple success rate calculation (can be improved with sliding window)
        current_rate = self.provider_success_rates[provider]
        self.provider_success_rates[provider] = (current_rate + (1.0 if success else 0.0)) / 2


class LLMService:
    """Main LLM service with provider management and fallback handling"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.metrics = LLMMetrics()
        self.cache: Dict[str, LLMResponse] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        try:
            # OpenAI
            if settings.openai_api_key:
                self.providers[LLMProvider.OPENAI] = create_provider(
                    LLMProvider.OPENAI, 
                    api_key=settings.openai_api_key
                )
            
            # Anthropic
            if settings.anthropic_api_key:
                self.providers[LLMProvider.ANTHROPIC] = create_provider(
                    LLMProvider.ANTHROPIC, 
                    api_key=settings.anthropic_api_key
                )
            
            # Mistral
            if settings.mistral_api_key:
                self.providers[LLMProvider.MISTRAL] = create_provider(
                    LLMProvider.MISTRAL, 
                    api_key=settings.mistral_api_key
                )
            
            # Venice.ai
            if settings.venice_ai_api_key:
                self.providers[LLMProvider.VENICE_AI] = create_provider(
                    LLMProvider.VENICE_AI, 
                    api_key=settings.venice_ai_api_key
                )
            
            # OpenRouter
            if settings.openrouter_api_key:
                self.providers[LLMProvider.OPENROUTER] = create_provider(
                    LLMProvider.OPENROUTER, 
                    api_key=settings.openrouter_api_key
                )
            
            # Ollama (always available if service is running)
            try:
                self.providers[LLMProvider.OLLAMA] = create_provider(
                    LLMProvider.OLLAMA, 
                    base_url=settings.ollama_base_url
                )
            except Exception as e:
                logger.info(f"Ollama not available: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing providers: {e}")
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_available_models(self, provider: Optional[LLMProvider] = None) -> Dict[LLMProvider, List[str]]:
        """Get available models for all or specific provider"""
        models = {}
        
        if provider:
            if provider in self.providers:
                models[provider] = [model.model_name for model in self.providers[provider].list_models()]
        else:
            for prov in self.providers:
                models[prov] = [model.model_name for model in self.providers[prov].list_models()]
        
        return models
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "messages": request.messages,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _should_use_fallback(self, error: Exception) -> bool:
        """Determine if fallback should be used based on error type"""
        # Rate limiting, service unavailable, etc.
        error_str = str(error).lower()
        fallback_triggers = [
            "rate limit",
            "service unavailable",
            "timeout",
            "connection error",
            "internal server error",
            "model not found"
        ]
        return any(trigger in error_str for trigger in fallback_triggers)
    
    def _get_fallback_provider(self, failed_provider: LLMProvider, attempt: int) -> Optional[LLMProvider]:
        """Get next fallback provider based on strategy"""
        if self.config.fallback_strategy == FallbackStrategy.NONE:
            return None
        
        available_providers = [p for p in self.config.fallback_providers if p in self.providers and p != failed_provider]
        
        if not available_providers or attempt >= len(available_providers):
            return None
        
        if self.config.fallback_strategy == FallbackStrategy.NEXT_PROVIDER:
            return available_providers[attempt]
        elif self.config.fallback_strategy == FallbackStrategy.CHEAPER_MODEL:
            # Sort by cost (ascending)
            sorted_providers = sorted(available_providers, key=lambda p: self._get_avg_cost(p))
            return sorted_providers[attempt] if attempt < len(sorted_providers) else None
        elif self.config.fallback_strategy == FallbackStrategy.FASTEST_MODEL:
            # Sort by success rate and response time
            sorted_providers = sorted(available_providers, key=lambda p: self._get_performance_score(p))
            return sorted_providers[attempt] if attempt < len(sorted_providers) else None
        elif self.config.fallback_strategy == FallbackStrategy.BEST_AVAILABLE:
            # Sort by overall quality score
            sorted_providers = sorted(available_providers, key=lambda p: self._get_quality_score(p), reverse=True)
            return sorted_providers[attempt] if attempt < len(sorted_providers) else None
        
        return None
    
    def _get_avg_cost(self, provider: LLMProvider) -> float:
        """Get average cost per token for provider"""
        if provider not in self.providers:
            return float('inf')
        
        models = self.providers[provider].list_models()
        if not models:
            return float('inf')
        
        return sum(model.cost_per_1k_tokens for model in models) / len(models)
    
    def _get_performance_score(self, provider: LLMProvider) -> float:
        """Get performance score for provider (lower is better)"""
        success_rate = self.metrics.provider_success_rates.get(provider, 0.5)
        # Combine success rate with response time (if available)
        return (1 - success_rate) * 1000  # Convert to milliseconds-like scale
    
    def _get_quality_score(self, provider: LLMProvider) -> float:
        """Get quality score for provider (higher is better)"""
        success_rate = self.metrics.provider_success_rates.get(provider, 0.5)
        # Simple quality score based on success rate
        return success_rate
    
    def _get_model_for_provider(self, provider: LLMProvider, preferred_model: str) -> str:
        """Get appropriate model for provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        available_models = self.providers[provider].list_models()
        model_names = [model.model_name for model in available_models]
        
        # Try to use preferred model if available
        if preferred_model in model_names:
            return preferred_model
        
        # Fallback to first available model
        if model_names:
            return model_names[0]
        
        raise ValueError(f"No models available for provider {provider}")
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with fallback handling"""
        
        # Use default model and provider if not specified
        target_model = model or self.config.primary_model
        target_provider = provider or self.config.primary_provider
        
        # Create request
        request = LLMRequest(
            messages=messages,
            model=target_model,
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            stop=kwargs.get('stop'),
            stream=kwargs.get('stream', False),
            functions=kwargs.get('functions'),
            metadata=kwargs.get('metadata', {})
        )
        
        # Check cache
        if self.config.enable_caching:
            cache_key = self._get_cache_key(request)
            if cache_key in self.cache:
                logger.info("Returning cached response")
                return self.cache[cache_key]
        
        # Try primary provider and fallbacks
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Get provider for this attempt
                if attempt == 0:
                    current_provider = target_provider
                else:
                    current_provider = self._get_fallback_provider(target_provider, attempt - 1)
                    if not current_provider:
                        break
                    self.metrics.fallback_uses += 1
                
                # Ensure provider is available
                if current_provider not in self.providers:
                    continue
                
                # Get appropriate model for this provider
                try:
                    current_model = self._get_model_for_provider(current_provider, target_model)
                    request.model = current_model
                except ValueError as e:
                    logger.warning(f"Model selection failed for {current_provider}: {e}")
                    continue
                
                # Check cost limit
                model_info = self.providers[current_provider].get_model_info(current_model)
                if model_info:
                    estimated_cost = (request.max_tokens / 1000) * model_info.cost_per_1k_tokens
                    if estimated_cost > self.config.cost_limit_per_request:
                        logger.warning(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${self.config.cost_limit_per_request:.4f}")
                        continue
                
                # Generate response
                start_time = datetime.now(timezone.utc)
                response = await asyncio.wait_for(
                    self.providers[current_provider].generate_response(request),
                    timeout=self.config.timeout_seconds
                )
                end_time = datetime.now(timezone.utc)
                
                # Update metrics
                response_time = (end_time - start_time).total_seconds() * 1000
                self.metrics.add_request(
                    current_provider, 
                    True, 
                    response.tokens_used, 
                    response.cost_usd, 
                    response_time
                )
                
                # Cache response
                if self.config.enable_caching:
                    self.cache[cache_key] = response
                
                logger.info(f"Generated response using {current_provider}/{current_model}")
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed with {current_provider}: {e}")
                
                # Update metrics
                self.metrics.add_request(current_provider, False, 0, 0, 0)
                
                # Check if we should use fallback
                if not self._should_use_fallback(e):
                    break
                
                # Wait before retry
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
        
        # All attempts failed
        logger.error(f"All LLM providers failed. Last error: {last_error}")
        raise RuntimeError(f"LLM service failed after {self.config.max_retries} retries: {last_error}")
    
    async def stream_response(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response with fallback handling"""
        
        if not self.config.enable_streaming:
            # Fallback to non-streaming
            response = await self.generate_response(messages, model, provider, **kwargs)
            yield response.content
            return
        
        # Use default model and provider if not specified
        target_model = model or self.config.primary_model
        target_provider = provider or self.config.primary_provider
        
        # Create request
        request = LLMRequest(
            messages=messages,
            model=target_model,
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            stop=kwargs.get('stop'),
            stream=True,
            functions=kwargs.get('functions'),
            metadata=kwargs.get('metadata', {})
        )
        
        # Try primary provider and fallbacks
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Get provider for this attempt
                if attempt == 0:
                    current_provider = target_provider
                else:
                    current_provider = self._get_fallback_provider(target_provider, attempt - 1)
                    if not current_provider:
                        break
                
                # Ensure provider is available
                if current_provider not in self.providers:
                    continue
                
                # Get appropriate model for this provider
                try:
                    current_model = self._get_model_for_provider(current_provider, target_model)
                    request.model = current_model
                except ValueError as e:
                    logger.warning(f"Model selection failed for {current_provider}: {e}")
                    continue
                
                # Check if model supports streaming
                model_info = self.providers[current_provider].get_model_info(current_model)
                if not model_info or not model_info.supports_streaming:
                    logger.warning(f"Model {current_model} doesn't support streaming")
                    continue
                
                # Stream response
                async for chunk in self.providers[current_provider].stream_response(request):
                    yield chunk
                
                logger.info(f"Streamed response using {current_provider}/{current_model}")
                return
                
            except Exception as e:
                last_error = e
                logger.warning(f"Streaming attempt {attempt + 1} failed with {current_provider}: {e}")
                
                # Check if we should use fallback
                if not self._should_use_fallback(e):
                    break
                
                # Wait before retry
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
        
        # All attempts failed, try non-streaming as final fallback
        logger.warning("Streaming failed, falling back to non-streaming")
        try:
            response = await self.generate_response(messages, model, provider, **kwargs)
            yield response.content
        except Exception as e:
            logger.error(f"Final fallback failed: {e}")
            raise RuntimeError(f"LLM service failed completely: {e}")
    
    async def health_check(self) -> Dict[LLMProvider, bool]:
        """Check health of all providers"""
        health_status = {}
        
        tasks = []
        for provider_name, provider in self.providers.items():
            tasks.append(self._check_provider_health(provider_name, provider))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (provider_name, _) in enumerate(self.providers.items()):
            health_status[provider_name] = results[i] if not isinstance(results[i], Exception) else False
        
        return health_status
    
    async def _check_provider_health(self, provider_name: LLMProvider, provider: BaseLLMProvider) -> bool:
        """Check individual provider health"""
        try:
            return await asyncio.wait_for(provider.health_check(), timeout=10.0)
        except Exception as e:
            logger.error(f"Health check failed for {provider_name}: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Union[int, float, Dict]]:
        """Get service metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "fallback_uses": self.metrics.fallback_uses,
            "total_tokens": self.metrics.total_tokens,
            "total_cost": self.metrics.total_cost,
            "avg_response_time": self.metrics.avg_response_time,
            "success_rate": self.metrics.successful_requests / max(self.metrics.total_requests, 1),
            "provider_success_rates": {k.value: v for k, v in self.metrics.provider_success_rates.items()},
            "available_providers": [p.value for p in self.get_available_providers()],
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        logger.info("Response cache cleared")
    
    def update_config(self, config: LLMConfig):
        """Update service configuration"""
        self.config = config
        logger.info("LLM service configuration updated")


# Global LLM service instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service