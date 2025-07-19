"""
Tests for Complete Chat Service

Comprehensive test suite for the integrated chat service including:
- Conversation management
- Response generation with tone injection
- Escalation handling
- End-to-end workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any

from app.services.chat_service import (
    ChatService, ChatRequest, ChatResponse, ClientConfiguration
)
from app.services.conversation_service import ConversationManager, ConversationContext, MessageRole
from app.services.response_service import (
    ResponseGenerator, GeneratedResponse, ResponseType, ToneStyle, ClientToneConfig
)
from app.services.escalation_service import EscalationManager, EscalationEvent, EscalationReason
from app.services.llm_service import LLMService


@pytest.fixture
def mock_llm_service():
    """Mock LLM service"""
    service = Mock(spec=LLMService)
    service.generate_response = AsyncMock(return_value=Mock(
        content="Hello! How can I help you today?",
        model="gpt-3.5-turbo",
        provider=Mock(value="openai"),
        tokens_used=25,
        cost_usd=0.00005,
        processing_time_ms=1000,
        confidence_score=0.9,
        metadata={"finish_reason": "stop"}
    ))
    service.get_metrics = Mock(return_value={
        "total_requests": 100,
        "successful_requests": 95,
        "available_providers": ["openai", "anthropic"]
    })
    return service


@pytest.fixture
def mock_conversation_manager():
    """Mock conversation manager"""
    manager = Mock(spec=ConversationManager)
    
    # Mock conversation context
    mock_context = Mock(spec=ConversationContext)
    mock_context.conversation_id = "conv_123"
    mock_context.client_id = "test_client"
    mock_context.session_id = "sess_456"
    mock_context.messages = []
    mock_context.add_message = Mock()
    
    manager.create_conversation = AsyncMock(return_value=mock_context)
    manager.get_conversation = AsyncMock(return_value=mock_context)
    manager.add_user_message = AsyncMock(return_value=mock_context)
    manager.add_assistant_message = AsyncMock(return_value=mock_context)
    manager.get_context_for_llm = AsyncMock(return_value=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ])
    manager.escalate_conversation = AsyncMock()
    manager.archive_conversation = AsyncMock()
    manager.get_conversation_stats = Mock(return_value={
        "active_conversations": 5,
        "total_messages": 100,
        "total_tokens": 5000
    })
    
    return manager


@pytest.fixture
def mock_response_generator():
    """Mock response generator"""
    generator = Mock(spec=ResponseGenerator)
    
    mock_response = GeneratedResponse(
        content="I'd be happy to help you with that!",
        response_type=ResponseType.ANSWER,
        confidence_score=0.85,
        tone_applied=ToneStyle.PROFESSIONAL,
        tokens_used=30,
        cost_usd=0.00006,
        processing_time_ms=800,
        should_escalate=False,
        metadata={"model_used": "gpt-3.5-turbo"}
    )
    
    generator.generate_response = AsyncMock(return_value=mock_response)
    generator.generate_greeting = AsyncMock(return_value="Hello! How can I help you today?")
    generator.generate_goodbye = AsyncMock(return_value="Thank you for contacting us!")
    
    return generator


@pytest.fixture
def mock_escalation_manager():
    """Mock escalation manager"""
    manager = Mock(spec=EscalationManager)
    manager.evaluate_escalation = AsyncMock(return_value=None)  # No escalation by default
    manager.resolve_escalation = AsyncMock()
    manager.get_active_escalations = Mock(return_value=[])
    manager.get_escalation_stats = Mock(return_value={
        "total_active": 0,
        "by_priority": {},
        "by_reason": {}
    })
    return manager


@pytest.fixture
def mock_search_service():
    """Mock search service"""
    service = Mock()
    service.hybrid_search = AsyncMock(return_value={
        "results": [
            {"content": "Relevant information about the user's query", "score": 0.8}
        ]
    })
    return service


@pytest.fixture
def mock_knowledge_base_service():
    """Mock knowledge base service"""
    service = Mock()
    return service


@pytest.fixture
def chat_service(
    mock_conversation_manager,
    mock_response_generator,
    mock_escalation_manager,
    mock_llm_service
):
    """Chat service with mocked dependencies"""
    service = ChatService(
        conversation_manager=mock_conversation_manager,
        response_generator=mock_response_generator,
        escalation_manager=mock_escalation_manager,
        llm_service=mock_llm_service
    )
    
    # Mock the search services
    service.search_service = Mock()
    service.search_service.hybrid_search = AsyncMock(return_value={
        "results": [{"content": "Test knowledge", "score": 0.8}]
    })
    service.knowledge_base_service = Mock()
    
    return service


class TestChatService:
    """Test suite for ChatService"""
    
    @pytest.mark.asyncio
    async def test_process_chat_message_basic(self, chat_service):
        """Test basic chat message processing"""
        request = ChatRequest(
            message="Hello, I need help",
            client_id="test_client",
            session_id="test_session"
        )
        
        response = await chat_service.process_chat_message(request)
        
        assert isinstance(response, ChatResponse)
        assert response.message == "I'd be happy to help you with that!"
        assert response.conversation_id == "conv_123"
        assert response.confidence_score == 0.85
        assert response.should_escalate is False
        assert response.tokens_used == 30
        assert response.cost_usd == 0.00006
    
    @pytest.mark.asyncio
    async def test_process_chat_message_with_existing_conversation(self, chat_service):
        """Test message processing with existing conversation"""
        request = ChatRequest(
            message="Follow-up question",
            client_id="test_client",
            session_id="test_session",
            conversation_id="conv_123"
        )
        
        response = await chat_service.process_chat_message(request)
        
        assert response.conversation_id == "conv_123"
        # Verify conversation manager was called to get existing conversation
        chat_service.conversation_manager.get_conversation.assert_called_with("conv_123")
    
    @pytest.mark.asyncio
    async def test_process_chat_message_with_escalation(
        self, chat_service, mock_escalation_manager
    ):
        """Test message processing that triggers escalation"""
        # Mock escalation event
        escalation_event = Mock(spec=EscalationEvent)
        escalation_event.to_dict.return_value = {
            "escalation_id": "esc_123",
            "reason": "user_request",
            "priority": "high"
        }
        mock_escalation_manager.evaluate_escalation.return_value = escalation_event
        
        request = ChatRequest(
            message="I want to speak to a human agent",
            client_id="test_client",
            session_id="test_session"
        )
        
        response = await chat_service.process_chat_message(request)
        
        assert response.should_escalate is True
        assert response.escalation_info is not None
        assert response.escalation_info["escalation_id"] == "esc_123"
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, chat_service):
        """Test starting a new conversation"""
        request = ChatRequest(
            message="",
            client_id="test_client",
            session_id="test_session"
        )
        
        response = await chat_service.start_conversation(request)
        
        assert response.message == "Hello! How can I help you today?"
        assert response.response_type == "greeting"
        assert response.confidence_score == 1.0
        assert response.should_escalate is False
        
        # Verify conversation was created
        chat_service.conversation_manager.create_conversation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_end_conversation(self, chat_service):
        """Test ending a conversation"""
        response = await chat_service.end_conversation("conv_123", "test_client")
        
        assert response.message == "Thank you for contacting us!"
        assert response.response_type == "goodbye"
        assert response.conversation_id == "conv_123"
        
        # Verify conversation was archived
        chat_service.conversation_manager.archive_conversation.assert_called_with("conv_123")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, chat_service, mock_response_generator):
        """Test error handling in message processing"""
        # Mock an exception in response generation
        mock_response_generator.generate_response.side_effect = Exception("LLM error")
        
        request = ChatRequest(
            message="Test message",
            client_id="test_client",
            session_id="test_session"
        )
        
        response = await chat_service.process_chat_message(request)
        
        assert response.should_escalate is True
        assert "technical difficulties" in response.message.lower()
        assert response.confidence_score == 0.0
        assert response.escalation_info["reason"] == "system_error"
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, chat_service):
        """Test getting conversation history"""
        # Mock conversation messages
        mock_messages = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            },
            {
                "role": "assistant", 
                "content": "Hi there!",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
        ]
        chat_service.conversation_manager.get_conversation.return_value.messages = mock_messages
        
        with patch.object(chat_service, 'get_conversation_history', return_value=mock_messages):
            history = await chat_service.get_conversation_history("conv_123")
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_update_client_configuration(self, chat_service):
        """Test updating client configuration"""
        tone_config = ClientToneConfig(
            primary_tone=ToneStyle.WARM,
            persona_description="Friendly helper"
        )
        
        client_config = ClientConfiguration(
            client_id="test_client",
            tone_config=tone_config
        )
        
        await chat_service.update_client_configuration("test_client", client_config)
        
        # Verify config was stored
        assert "test_client" in chat_service.client_configs
        assert chat_service.client_configs["test_client"].tone_config.primary_tone == ToneStyle.WARM
    
    @pytest.mark.asyncio
    async def test_knowledge_base_search_integration(self, chat_service):
        """Test knowledge base search integration"""
        request = ChatRequest(
            message="How do I reset my password?",
            client_id="test_client",
            session_id="test_session"
        )
        
        # Mock client config with knowledge base
        client_config = ClientConfiguration(
            client_id="test_client",
            tone_config=ClientToneConfig(),
            knowledge_base_ids=["kb_general"]
        )
        chat_service.client_configs["test_client"] = client_config
        
        response = await chat_service.process_chat_message(request)
        
        # Verify search was called
        chat_service.search_service.hybrid_search.assert_called_once()
        assert response.metadata["knowledge_context_found"] is True
    
    def test_get_service_stats(self, chat_service):
        """Test getting service statistics"""
        stats = chat_service.get_service_stats()
        
        assert "conversation_stats" in stats
        assert "escalation_stats" in stats
        assert "llm_metrics" in stats
        assert "active_clients" in stats
        assert "timestamp" in stats
        
        assert stats["conversation_stats"]["active_conversations"] == 5
        assert stats["escalation_stats"]["total_active"] == 0
        assert len(stats["llm_metrics"]["available_providers"]) == 2


class TestChatServiceIntegration:
    """Integration tests for chat service"""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, chat_service):
        """Test complete conversation flow from start to end"""
        client_id = "test_client"
        session_id = "test_session"
        
        # Start conversation
        start_request = ChatRequest(
            message="",
            client_id=client_id,
            session_id=session_id
        )
        start_response = await chat_service.start_conversation(start_request)
        conversation_id = start_response.conversation_id
        
        # Send message
        message_request = ChatRequest(
            message="I need help with my account",
            client_id=client_id,
            session_id=session_id,
            conversation_id=conversation_id
        )
        message_response = await chat_service.process_chat_message(message_request)
        
        # End conversation
        end_response = await chat_service.end_conversation(conversation_id, client_id)
        
        # Verify flow
        assert start_response.response_type == "greeting"
        assert message_response.conversation_id == conversation_id
        assert end_response.response_type == "goodbye"
        assert end_response.conversation_id == conversation_id
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, chat_service):
        """Test multi-turn conversation handling"""
        request_base = ChatRequest(
            message="",
            client_id="test_client",
            session_id="test_session"
        )
        
        # Start conversation
        start_response = await chat_service.start_conversation(request_base)
        conversation_id = start_response.conversation_id
        
        # Multiple message exchanges
        messages = [
            "I have a problem with my order",
            "My order number is 12345",
            "I need to change the delivery address"
        ]
        
        responses = []
        for message in messages:
            request = ChatRequest(
                message=message,
                client_id="test_client",
                session_id="test_session",
                conversation_id=conversation_id
            )
            response = await chat_service.process_chat_message(request)
            responses.append(response)
        
        # Verify all responses use same conversation
        for response in responses:
            assert response.conversation_id == conversation_id
        
        # Verify conversation manager was called to add messages
        assert chat_service.conversation_manager.add_user_message.call_count == len(messages)
        assert chat_service.conversation_manager.add_assistant_message.call_count == len(messages)


class TestChatServiceConfiguration:
    """Test configuration handling"""
    
    @pytest.mark.asyncio
    async def test_tone_application(self, chat_service):
        """Test that client tone configuration is applied"""
        # Configure warm tone
        tone_config = ClientToneConfig(
            primary_tone=ToneStyle.WARM,
            persona_description="Friendly customer success representative"
        )
        
        client_config = ClientConfiguration(
            client_id="test_client",
            tone_config=tone_config
        )
        
        await chat_service.update_client_configuration("test_client", client_config)
        
        request = ChatRequest(
            message="Hello",
            client_id="test_client",
            session_id="test_session"
        )
        
        response = await chat_service.process_chat_message(request)
        
        # Verify tone was applied (check response generator was called with correct config)
        call_args = chat_service.response_generator.generate_response.call_args[0][0]
        assert call_args.client_config.primary_tone == ToneStyle.WARM
    
    @pytest.mark.asyncio
    async def test_escalation_configuration(self, chat_service, mock_escalation_manager):
        """Test custom escalation configuration"""
        # Configure custom escalation triggers
        client_config = ClientConfiguration(
            client_id="test_client",
            tone_config=ClientToneConfig(),
            escalation_config={"custom_trigger": "billing dispute"}
        )
        
        await chat_service.update_client_configuration("test_client", client_config)
        
        request = ChatRequest(
            message="I have a billing dispute",
            client_id="test_client",
            session_id="test_session"
        )
        
        await chat_service.process_chat_message(request)
        
        # Verify escalation manager was called with client config
        mock_escalation_manager.evaluate_escalation.assert_called_once()
        call_args = mock_escalation_manager.evaluate_escalation.call_args
        assert call_args[0][3] == {"custom_trigger": "billing dispute"}  # escalation config