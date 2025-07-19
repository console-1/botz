"""
LLM API Endpoints

Provides REST API endpoints for interacting with the LLM service.
Supports multiple providers, model selection, and streaming responses.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import json
import asyncio
from datetime import datetime

from ...services.llm_service import get_llm_service, LLMService, LLMConfig, FallbackStrategy
from ...services.llm_providers import LLMProvider, LLMModelType
from ...core.config import settings

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message schema"""
    role: str = Field(..., description="Role of the message sender", example="user")
    content: str = Field(..., description="Content of the message", example="Hello, how are you?")


class ChatRequest(BaseModel):
    """Chat completion request schema"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Model to use", example="gpt-3.5-turbo")
    provider: Optional[LLMProvider] = Field(None, description="LLM provider to use")
    max_tokens: int = Field(1000, description="Maximum tokens in response", ge=1, le=8192)
    temperature: float = Field(0.7, description="Temperature for response generation", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Top-p sampling parameter", ge=0.0, le=1.0)
    stream: bool = Field(False, description="Enable streaming response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    functions: Optional[List[Dict[str, Any]]] = Field(None, description="Functions for function calling")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "model": "gpt-3.5-turbo",
                "provider": "openai",
                "max_tokens": 500,
                "temperature": 0.7,
                "stream": False
            }
        }


class ChatResponse(BaseModel):
    """Chat completion response schema"""
    content: str = Field(..., description="Generated response content")
    model: str = Field(..., description="Model used for generation")
    provider: str = Field(..., description="Provider used for generation")
    tokens_used: int = Field(..., description="Total tokens used")
    cost_usd: float = Field(..., description="Cost in USD")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "The capital of France is Paris.",
                "model": "gpt-3.5-turbo",
                "provider": "openai",
                "tokens_used": 25,
                "cost_usd": 0.00005,
                "processing_time_ms": 1250,
                "confidence_score": 0.95,
                "metadata": {"finish_reason": "stop"}
            }
        }


class ProvidersResponse(BaseModel):
    """Available providers response schema"""
    providers: List[str] = Field(..., description="List of available providers")
    
    class Config:
        schema_extra = {
            "example": {
                "providers": ["openai", "anthropic", "mistral", "venice_ai", "openrouter", "ollama"]
            }
        }


class ModelsResponse(BaseModel):
    """Available models response schema"""
    models: Dict[str, List[str]] = Field(..., description="Models grouped by provider")
    
    class Config:
        schema_extra = {
            "example": {
                "models": {
                    "openai": ["gpt-4", "gpt-3.5-turbo"],
                    "anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku"],
                    "mistral": ["mistral-large", "mistral-small"]
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Overall health status")
    providers: Dict[str, bool] = Field(..., description="Health status per provider")
    timestamp: datetime = Field(..., description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "providers": {
                    "openai": True,
                    "anthropic": True,
                    "mistral": False
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class MetricsResponse(BaseModel):
    """Service metrics response schema"""
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    fallback_uses: int = Field(..., description="Number of fallback uses")
    total_tokens: int = Field(..., description="Total tokens processed")
    total_cost: float = Field(..., description="Total cost in USD")
    avg_response_time: float = Field(..., description="Average response time in milliseconds")
    success_rate: float = Field(..., description="Success rate as percentage")
    provider_success_rates: Dict[str, float] = Field(..., description="Success rate per provider")
    available_providers: List[str] = Field(..., description="Available providers")
    cache_size: int = Field(..., description="Current cache size")
    
    class Config:
        schema_extra = {
            "example": {
                "total_requests": 1250,
                "successful_requests": 1200,
                "failed_requests": 50,
                "fallback_uses": 25,
                "total_tokens": 125000,
                "total_cost": 12.50,
                "avg_response_time": 1250.0,
                "success_rate": 0.96,
                "provider_success_rates": {
                    "openai": 0.98,
                    "anthropic": 0.95,
                    "mistral": 0.92
                },
                "available_providers": ["openai", "anthropic", "mistral"],
                "cache_size": 100
            }
        }


class ConfigUpdateRequest(BaseModel):
    """Configuration update request schema"""
    primary_provider: Optional[LLMProvider] = Field(None, description="Primary provider")
    primary_model: Optional[str] = Field(None, description="Primary model")
    fallback_strategy: Optional[FallbackStrategy] = Field(None, description="Fallback strategy")
    fallback_providers: Optional[List[LLMProvider]] = Field(None, description="Fallback providers")
    max_retries: Optional[int] = Field(None, description="Maximum retries", ge=0, le=10)
    retry_delay: Optional[float] = Field(None, description="Retry delay in seconds", ge=0.0, le=60.0)
    timeout_seconds: Optional[float] = Field(None, description="Timeout in seconds", ge=1.0, le=300.0)
    enable_streaming: Optional[bool] = Field(None, description="Enable streaming")
    enable_caching: Optional[bool] = Field(None, description="Enable caching")
    cost_limit_per_request: Optional[float] = Field(None, description="Cost limit per request in USD", ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "primary_provider": "openai",
                "primary_model": "gpt-3.5-turbo",
                "fallback_strategy": "next_provider",
                "max_retries": 3,
                "enable_streaming": True,
                "enable_caching": True,
                "cost_limit_per_request": 0.10
            }
        }


def get_llm_service_dependency() -> LLMService:
    """Dependency to get LLM service instance"""
    return get_llm_service()


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """
    Generate chat completion using LLM service
    
    Supports multiple providers with automatic fallback handling.
    """
    try:
        # Convert request to messages format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate response
        response = await llm_service.generate_response(
            messages=messages,
            model=request.model,
            provider=request.provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            functions=request.functions,
            metadata=request.metadata
        )
        
        return ChatResponse(
            content=response.content,
            model=response.model,
            provider=response.provider.value,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            processing_time_ms=response.processing_time_ms,
            confidence_score=response.confidence_score,
            metadata=response.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@router.post("/chat/completions/stream")
async def chat_completions_stream(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """
    Generate streaming chat completion using LLM service
    
    Returns Server-Sent Events (SSE) stream of response chunks.
    """
    try:
        # Convert request to messages format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        async def generate():
            async for chunk in llm_service.stream_response(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                functions=request.functions,
                metadata=request.metadata
            ):
                # Format as SSE
                yield f"data: {json.dumps({'content': chunk, 'done': False})}\\n\\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'content': '', 'done': True})}\\n\\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming chat completion failed: {str(e)}")


@router.get("/providers", response_model=ProvidersResponse)
async def get_providers(
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Get list of available LLM providers"""
    try:
        providers = llm_service.get_available_providers()
        return ProvidersResponse(providers=[p.value for p in providers])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")


@router.get("/models", response_model=ModelsResponse)
async def get_models(
    provider: Optional[LLMProvider] = Query(None, description="Filter by provider"),
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Get list of available models for all or specific provider"""
    try:
        models = llm_service.get_available_models(provider)
        return ModelsResponse(models={p.value: models for p, models in models.items()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Check health of all LLM providers"""
    try:
        health_status = await llm_service.health_check()
        overall_status = "healthy" if any(health_status.values()) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            providers={p.value: status for p, status in health_status.items()},
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Get service metrics and performance statistics"""
    try:
        metrics = llm_service.get_metrics()
        return MetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/config")
async def update_config(
    config_request: ConfigUpdateRequest,
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Update LLM service configuration"""
    try:
        # Get current config
        current_config = llm_service.config
        
        # Update only provided fields
        config_dict = current_config.__dict__.copy()
        for field, value in config_request.dict(exclude_unset=True).items():
            config_dict[field] = value
        
        # Create new config
        new_config = LLMConfig(**config_dict)
        
        # Update service
        llm_service.update_config(new_config)
        
        return {"message": "Configuration updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/cache/clear")
async def clear_cache(
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Clear response cache"""
    try:
        llm_service.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/models/info/{provider}")
async def get_model_info(
    provider: LLMProvider,
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Get detailed information about models for a specific provider"""
    try:
        if provider not in llm_service.providers:
            raise HTTPException(status_code=404, detail=f"Provider {provider} not available")
        
        models = llm_service.providers[provider].list_models()
        
        return {
            "provider": provider.value,
            "models": [
                {
                    "name": model.model_name,
                    "type": model.model_type.value,
                    "max_tokens": model.max_tokens,
                    "cost_per_1k_tokens": model.cost_per_1k_tokens,
                    "supports_streaming": model.supports_streaming,
                    "supports_function_calling": model.supports_function_calling,
                    "context_window": model.context_window
                }
                for model in models
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.get("/config")
async def get_config(
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Get current LLM service configuration"""
    try:
        config = llm_service.config
        return {
            "primary_provider": config.primary_provider.value,
            "primary_model": config.primary_model,
            "fallback_strategy": config.fallback_strategy.value,
            "fallback_providers": [p.value for p in config.fallback_providers],
            "max_retries": config.max_retries,
            "retry_delay": config.retry_delay,
            "timeout_seconds": config.timeout_seconds,
            "enable_streaming": config.enable_streaming,
            "enable_caching": config.enable_caching,
            "cost_limit_per_request": config.cost_limit_per_request
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@router.post("/test")
async def test_llm_service(
    provider: Optional[LLMProvider] = Query(None, description="Test specific provider"),
    model: Optional[str] = Query(None, description="Test specific model"),
    llm_service: LLMService = Depends(get_llm_service_dependency)
):
    """Test LLM service with a simple request"""
    try:
        test_messages = [
            {"role": "user", "content": "Hello, can you respond with just 'Hello back!' please?"}
        ]
        
        response = await llm_service.generate_response(
            messages=test_messages,
            model=model,
            provider=provider,
            max_tokens=20,
            temperature=0.1
        )
        
        return {
            "status": "success",
            "response": response.content,
            "model": response.model,
            "provider": response.provider.value,
            "tokens_used": response.tokens_used,
            "cost_usd": response.cost_usd,
            "processing_time_ms": response.processing_time_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")