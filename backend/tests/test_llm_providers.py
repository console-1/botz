"""
Tests for LLM Providers

Comprehensive test suite for testing LLM provider implementations,
including OpenAI, Anthropic, Mistral, Venice.ai, OpenRouter, and Ollama.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any

from app.services.llm_providers import (
    LLMProvider, LLMModelType, LLMModel, LLMRequest, LLMResponse,
    BaseLLMProvider, OpenAIProvider, AnthropicProvider, OllamaProvider,
    GenericAPIProvider, create_provider, LLM_MODELS
)


class TestLLMModels:
    """Test suite for LLM model configurations"""
    
    def test_llm_model_creation(self):
        """Test LLM model creation"""
        model = LLMModel(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            model_type=LLMModelType.CHAT,
            max_tokens=4096,
            cost_per_1k_tokens=0.001,
            supports_streaming=True,
            supports_function_calling=True,
            context_window=16385
        )
        
        assert model.provider == LLMProvider.OPENAI
        assert model.model_name == "gpt-3.5-turbo"
        assert model.model_type == LLMModelType.CHAT
        assert model.max_tokens == 4096
        assert model.cost_per_1k_tokens == 0.001
        assert model.supports_streaming is True
        assert model.supports_function_calling is True
        assert model.context_window == 16385
        assert str(model) == "openai/gpt-3.5-turbo"
    
    def test_llm_models_configuration(self):
        """Test that all providers have models configured"""
        for provider in LLMProvider:
            assert provider in LLM_MODELS
            assert len(LLM_MODELS[provider]) > 0
            
            for model_name, model in LLM_MODELS[provider].items():
                assert isinstance(model, LLMModel)
                assert model.provider == provider
                assert model.model_name == model_name
                assert model.cost_per_1k_tokens >= 0
                assert model.max_tokens > 0
                assert model.context_window > 0


class TestLLMRequest:
    """Test suite for LLM request objects"""
    
    def test_llm_request_creation(self):
        """Test LLM request creation"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        request = LLMRequest(
            messages=messages,
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stop=["\\n"],
            stream=False,
            functions=None,
            metadata={"test": "value"}
        )
        
        assert request.messages == messages
        assert request.model == "gpt-3.5-turbo"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stop == ["\\n"]
        assert request.stream is False
        assert request.functions is None
        assert request.metadata == {"test": "value"}
    
    def test_llm_request_defaults(self):
        """Test LLM request with default values"""
        messages = [{"role": "user", "content": "Hello"}]
        request = LLMRequest(messages=messages, model="gpt-3.5-turbo")
        
        assert request.max_tokens == 1000
        assert request.temperature == 0.7
        assert request.top_p == 1.0
        assert request.stop is None
        assert request.stream is False
        assert request.functions is None
        assert request.metadata == {}


class TestLLMResponse:
    """Test suite for LLM response objects"""
    
    def test_llm_response_creation(self):
        """Test LLM response creation"""
        response = LLMResponse(
            content="Hello back!",
            model="gpt-3.5-turbo",
            provider=LLMProvider.OPENAI,
            tokens_used=10,
            cost_usd=0.00001,
            processing_time_ms=1250,
            confidence_score=0.95,
            metadata={"finish_reason": "stop"}
        )
        
        assert response.content == "Hello back!"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == LLMProvider.OPENAI
        assert response.tokens_used == 10
        assert response.cost_usd == 0.00001
        assert response.processing_time_ms == 1250
        assert response.confidence_score == 0.95
        assert response.metadata == {"finish_reason": "stop"}
    
    def test_llm_response_defaults(self):
        """Test LLM response with default values"""
        response = LLMResponse(
            content="Hello",
            model="gpt-3.5-turbo",
            provider=LLMProvider.OPENAI,
            tokens_used=5,
            cost_usd=0.000005,
            processing_time_ms=800
        )
        
        assert response.confidence_score is None
        assert response.metadata == {}


class TestBaseLLMProvider:
    """Test suite for base LLM provider"""
    
    def test_base_provider_initialization(self):
        """Test base provider initialization"""
        provider = BaseLLMProvider(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            base_url="https://api.test.com"
        )
        
        assert provider.provider == LLMProvider.OPENAI
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://api.test.com"
        assert provider.client is None
        assert provider.models == LLM_MODELS[LLMProvider.OPENAI]
    
    def test_get_model_info(self):
        """Test getting model information"""
        provider = BaseLLMProvider(LLMProvider.OPENAI)
        
        model_info = provider.get_model_info("gpt-3.5-turbo")
        assert model_info is not None
        assert model_info.model_name == "gpt-3.5-turbo"
        assert model_info.provider == LLMProvider.OPENAI
        
        # Test non-existent model
        model_info = provider.get_model_info("non-existent")
        assert model_info is None
    
    def test_list_models(self):
        """Test listing available models"""
        provider = BaseLLMProvider(LLMProvider.OPENAI)
        
        models = provider.list_models()
        assert len(models) > 0
        assert all(isinstance(model, LLMModel) for model in models)
        assert all(model.provider == LLMProvider.OPENAI for model in models)
    
    def test_calculate_cost(self):
        """Test cost calculation"""
        provider = BaseLLMProvider(LLMProvider.OPENAI)
        
        # Test with existing model
        cost = provider.calculate_cost(1000, "gpt-3.5-turbo")
        expected_cost = LLM_MODELS[LLMProvider.OPENAI]["gpt-3.5-turbo"].cost_per_1k_tokens
        assert cost == expected_cost
        
        # Test with non-existent model
        cost = provider.calculate_cost(1000, "non-existent")
        assert cost == 0.0


class TestOpenAIProvider:
    """Test suite for OpenAI provider"""
    
    @pytest.fixture
    def openai_provider(self):
        """OpenAI provider fixture"""
        return OpenAIProvider(api_key="test-key")
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello back!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 10
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5
        return mock_response
    
    @pytest.mark.asyncio
    async def test_generate_response(self, openai_provider, mock_openai_response):
        """Test OpenAI response generation"""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )
        
        with patch.object(openai_provider.client.chat.completions, 'create', return_value=mock_openai_response):
            response = await openai_provider.generate_response(request)
            
            assert response.content == "Hello back!"
            assert response.model == "gpt-3.5-turbo"
            assert response.provider == LLMProvider.OPENAI
            assert response.tokens_used == 10
            assert response.cost_usd > 0
            assert response.processing_time_ms > 0
            assert response.metadata["finish_reason"] == "stop"
    
    @pytest.mark.asyncio
    async def test_generate_response_error(self, openai_provider):
        """Test OpenAI response generation error handling"""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )
        
        with patch.object(openai_provider.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await openai_provider.generate_response(request)
    
    @pytest.mark.asyncio
    async def test_stream_response(self, openai_provider):
        """Test OpenAI streaming response"""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )
        
        # Mock streaming response
        async def mock_stream():
            for chunk in ["Hello", " back", "!"]:
                mock_chunk = Mock()
                mock_chunk.choices = [Mock()]
                mock_chunk.choices[0].delta.content = chunk
                yield mock_chunk
        
        with patch.object(openai_provider.client.chat.completions, 'create', return_value=mock_stream()):
            chunks = []
            async for chunk in openai_provider.stream_response(request):
                chunks.append(chunk)
            
            assert chunks == ["Hello", " back", "!"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, openai_provider, mock_openai_response):
        """Test OpenAI health check"""
        with patch.object(openai_provider.client.chat.completions, 'create', return_value=mock_openai_response):
            health = await openai_provider.health_check()
            assert health is True
        
        with patch.object(openai_provider.client.chat.completions, 'create', side_effect=Exception("Error")):
            health = await openai_provider.health_check()
            assert health is False


class TestAnthropicProvider:
    """Test suite for Anthropic provider"""
    
    @pytest.fixture
    def anthropic_provider(self):
        """Anthropic provider fixture"""
        return AnthropicProvider(api_key="test-key")
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API response"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Hello back!"
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 5
        return mock_response
    
    @pytest.mark.asyncio
    async def test_generate_response(self, anthropic_provider, mock_anthropic_response):
        """Test Anthropic response generation"""
        request = LLMRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            model="claude-3-5-haiku-20241022"
        )
        
        with patch.object(anthropic_provider.client.messages, 'create', return_value=mock_anthropic_response):
            response = await anthropic_provider.generate_response(request)
            
            assert response.content == "Hello back!"
            assert response.model == "claude-3-5-haiku-20241022"
            assert response.provider == LLMProvider.ANTHROPIC
            assert response.tokens_used == 10
            assert response.cost_usd > 0
            assert response.processing_time_ms > 0
            assert response.metadata["stop_reason"] == "end_turn"
    
    @pytest.mark.asyncio
    async def test_health_check(self, anthropic_provider, mock_anthropic_response):
        """Test Anthropic health check"""
        with patch.object(anthropic_provider.client.messages, 'create', return_value=mock_anthropic_response):
            health = await anthropic_provider.health_check()
            assert health is True
        
        with patch.object(anthropic_provider.client.messages, 'create', side_effect=Exception("Error")):
            health = await anthropic_provider.health_check()
            assert health is False


class TestOllamaProvider:
    """Test suite for Ollama provider"""
    
    @pytest.fixture
    def ollama_provider(self):
        """Ollama provider fixture"""
        return OllamaProvider(base_url="http://localhost:11434")
    
    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama API response"""
        return {
            "message": {"content": "Hello back!"},
            "eval_count": 5,
            "prompt_eval_count": 5,
            "eval_duration": 1000000000  # 1 second in nanoseconds
        }
    
    @pytest.mark.asyncio
    async def test_generate_response(self, ollama_provider, mock_ollama_response):
        """Test Ollama response generation"""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama3"
        )
        
        with patch.object(ollama_provider.client, 'chat', return_value=mock_ollama_response):
            response = await ollama_provider.generate_response(request)
            
            assert response.content == "Hello back!"
            assert response.model == "llama3"
            assert response.provider == LLMProvider.OLLAMA
            assert response.tokens_used == 10
            assert response.cost_usd == 0.0  # Free for self-hosted
            assert response.processing_time_ms > 0
            assert response.metadata["eval_count"] == 5
    
    @pytest.mark.asyncio
    async def test_health_check(self, ollama_provider):
        """Test Ollama health check"""
        mock_models = {"models": [{"name": "llama3"}]}
        
        with patch.object(ollama_provider.client, 'list', return_value=mock_models):
            health = await ollama_provider.health_check()
            assert health is True
        
        with patch.object(ollama_provider.client, 'list', side_effect=Exception("Error")):
            health = await ollama_provider.health_check()
            assert health is False


class TestGenericAPIProvider:
    """Test suite for generic API provider"""
    
    @pytest.fixture
    def generic_provider(self):
        """Generic API provider fixture"""
        return GenericAPIProvider(
            provider=LLMProvider.VENICE_AI,
            api_key="test-key",
            base_url="https://api.venice.ai/v1"
        )
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock generic API response"""
        return {
            "choices": [{
                "message": {"content": "Hello back!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "total_tokens": 10,
                "prompt_tokens": 5,
                "completion_tokens": 5
            }
        }
    
    @pytest.mark.asyncio
    async def test_generate_response(self, generic_provider, mock_api_response):
        """Test generic API response generation"""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3-8b-instruct"
        )
        
        with patch.object(generic_provider.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_api_response
            mock_post.return_value = mock_response
            
            response = await generic_provider.generate_response(request)
            
            assert response.content == "Hello back!"
            assert response.model == "llama-3-8b-instruct"
            assert response.provider == LLMProvider.VENICE_AI
            assert response.tokens_used == 10
            assert response.cost_usd > 0
            assert response.processing_time_ms > 0
            assert response.metadata["finish_reason"] == "stop"
    
    @pytest.mark.asyncio
    async def test_health_check(self, generic_provider):
        """Test generic API health check"""
        with patch.object(generic_provider.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            health = await generic_provider.health_check()
            assert health is True
        
        with patch.object(generic_provider.client, 'get', side_effect=Exception("Error")):
            health = await generic_provider.health_check()
            assert health is False


class TestProviderFactory:
    """Test suite for provider factory"""
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider"""
        provider = create_provider(LLMProvider.OPENAI, api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider == LLMProvider.OPENAI
    
    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider"""
        provider = create_provider(LLMProvider.ANTHROPIC, api_key="test-key")
        assert isinstance(provider, AnthropicProvider)
        assert provider.provider == LLMProvider.ANTHROPIC
    
    def test_create_ollama_provider(self):
        """Test creating Ollama provider"""
        provider = create_provider(LLMProvider.OLLAMA, base_url="http://localhost:11434")
        assert isinstance(provider, OllamaProvider)
        assert provider.provider == LLMProvider.OLLAMA
    
    def test_create_mistral_provider(self):
        """Test creating Mistral provider"""
        provider = create_provider(LLMProvider.MISTRAL, api_key="test-key")
        assert isinstance(provider, GenericAPIProvider)
        assert provider.provider == LLMProvider.MISTRAL
    
    def test_create_venice_ai_provider(self):
        """Test creating Venice.ai provider"""
        provider = create_provider(LLMProvider.VENICE_AI, api_key="test-key")
        assert isinstance(provider, GenericAPIProvider)
        assert provider.provider == LLMProvider.VENICE_AI
    
    def test_create_openrouter_provider(self):
        """Test creating OpenRouter provider"""
        provider = create_provider(LLMProvider.OPENROUTER, api_key="test-key")
        assert isinstance(provider, GenericAPIProvider)
        assert provider.provider == LLMProvider.OPENROUTER
    
    def test_create_unsupported_provider(self):
        """Test creating unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider("unsupported")


class TestProviderEnums:
    """Test suite for provider enums"""
    
    def test_llm_provider_enum(self):
        """Test LLM provider enum values"""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.MISTRAL == "mistral"
        assert LLMProvider.VENICE_AI == "venice_ai"
        assert LLMProvider.OPENROUTER == "openrouter"
        assert LLMProvider.OLLAMA == "ollama"
    
    def test_llm_model_type_enum(self):
        """Test LLM model type enum values"""
        assert LLMModelType.CHAT == "chat"
        assert LLMModelType.COMPLETION == "completion"
        assert LLMModelType.EMBEDDING == "embedding"
        assert LLMModelType.FUNCTION_CALLING == "function_calling"


if __name__ == "__main__":
    pytest.main([__file__])