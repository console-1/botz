"""
Complete Chat API Endpoints

Comprehensive REST API for the customer service bot with full conversation management,
tone injection, escalation handling, and quality control.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import json
import asyncio
from datetime import datetime

from ...services.chat_service import get_chat_service, ChatService, ChatRequest, ChatResponse, ClientConfiguration
from ...services.conversation_service import ConversationState
from ...services.response_service import ToneStyle, ClientToneConfig
from ...services.escalation_service import EscalationPriority, EscalationReason
from ...core.config import settings

router = APIRouter()


# Request/Response Models

class MessageRequest(BaseModel):
    """Chat message request"""
    message: str = Field(..., description="User message", example="I need help with my order")
    client_id: str = Field(..., description="Client identifier", example="acme-corp")
    session_id: str = Field(..., description="Session identifier", example="sess_123456")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "I need help with my order status",
                "client_id": "acme-corp",
                "session_id": "sess_123456",
                "conversation_id": "conv_abc123",
                "user_id": "user_789",
                "metadata": {"user_tier": "premium", "source": "mobile_app"}
            }
        }


class MessageResponse(BaseModel):
    """Chat message response"""
    message: str = Field(..., description="Bot response")
    conversation_id: str = Field(..., description="Conversation identifier")
    response_type: str = Field(..., description="Type of response")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    should_escalate: bool = Field(..., description="Whether escalation is recommended")
    escalation_info: Optional[Dict[str, Any]] = Field(None, description="Escalation details if applicable")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    tokens_used: int = Field(..., description="Tokens used for generation")
    cost_usd: float = Field(..., description="Cost in USD")
    metadata: Dict[str, Any] = Field(..., description="Additional response metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "I'd be happy to help you check your order status. Could you please provide your order number?",
                "conversation_id": "conv_abc123",
                "response_type": "clarification",
                "confidence_score": 0.85,
                "should_escalate": False,
                "escalation_info": None,
                "processing_time_ms": 1250,
                "tokens_used": 45,
                "cost_usd": 0.00009,
                "metadata": {
                    "tone_applied": "professional",
                    "knowledge_context_found": False,
                    "conversation_length": 3
                }
            }
        }


class ConversationStartRequest(BaseModel):
    """Start conversation request"""
    client_id: str = Field(..., description="Client identifier")
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Initial metadata")


class ClientConfigRequest(BaseModel):
    """Client configuration request"""
    primary_tone: ToneStyle = Field(ToneStyle.PROFESSIONAL, description="Primary tone style")
    secondary_tone: Optional[ToneStyle] = Field(None, description="Secondary tone style")
    persona_description: str = Field("A helpful customer service representative", description="Bot persona")
    brand_voice: Optional[str] = Field(None, description="Brand voice description")
    custom_instructions: List[str] = Field(default_factory=list, description="Custom instructions")
    prohibited_words: List[str] = Field(default_factory=list, description="Words to avoid")
    preferred_phrases: Dict[str, str] = Field(default_factory=dict, description="Phrase substitutions")
    escalation_triggers: List[str] = Field(default_factory=list, description="Custom escalation triggers")
    max_response_length: int = Field(500, description="Maximum response length")
    knowledge_base_ids: List[str] = Field(default_factory=list, description="Knowledge base IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "primary_tone": "warm",
                "secondary_tone": "professional",
                "persona_description": "A friendly and knowledgeable customer success representative",
                "brand_voice": "We are approachable, helpful, and always put customers first",
                "custom_instructions": [
                    "Always offer to help with related questions",
                    "Mention our 24/7 support if escalating"
                ],
                "prohibited_words": ["cheap", "expensive"],
                "preferred_phrases": {
                    "problem": "challenge",
                    "issue": "concern"
                },
                "escalation_triggers": ["billing dispute", "legal question"],
                "max_response_length": 300,
                "knowledge_base_ids": ["kb_general", "kb_policies"]
            }
        }


class ConversationHistoryResponse(BaseModel):
    """Conversation history response"""
    conversation_id: str = Field(..., description="Conversation identifier")
    messages: List[Dict[str, Any]] = Field(..., description="Conversation messages")
    total_messages: int = Field(..., description="Total message count")
    conversation_state: str = Field(..., description="Current conversation state")
    created_at: datetime = Field(..., description="Conversation creation time")
    updated_at: datetime = Field(..., description="Last update time")


class EscalationResponse(BaseModel):
    """Escalation response"""
    escalation_id: str = Field(..., description="Escalation identifier")
    conversation_id: str = Field(..., description="Related conversation ID")
    reason: str = Field(..., description="Escalation reason")
    priority: str = Field(..., description="Escalation priority")
    confidence_score: float = Field(..., description="Original confidence score")
    sentiment_score: str = Field(..., description="Detected sentiment")
    created_at: datetime = Field(..., description="Escalation creation time")
    status: str = Field(..., description="Escalation status")


class ServiceStatsResponse(BaseModel):
    """Service statistics response"""
    conversation_stats: Dict[str, Any] = Field(..., description="Conversation statistics")
    escalation_stats: Dict[str, Any] = Field(..., description="Escalation statistics")
    llm_metrics: Dict[str, Any] = Field(..., description="LLM service metrics")
    active_clients: int = Field(..., description="Number of active clients")
    timestamp: str = Field(..., description="Statistics timestamp")


# API Endpoints

def get_chat_service_dependency() -> ChatService:
    """Dependency to get chat service instance"""
    return get_chat_service()


@router.post("/message", response_model=MessageResponse)
async def send_message(
    request: MessageRequest,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """
    Send a message and get bot response
    
    Processes user message through complete pipeline:
    - Conversation management
    - Knowledge base search
    - Response generation with tone injection
    - Quality control and escalation evaluation
    """
    try:
        chat_request = ChatRequest(
            message=request.message,
            client_id=request.client_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            metadata=request.metadata or {}
        )
        
        response = await chat_service.process_chat_message(chat_request)
        
        return MessageResponse(
            message=response.message,
            conversation_id=response.conversation_id,
            response_type=response.response_type,
            confidence_score=response.confidence_score,
            should_escalate=response.should_escalate,
            escalation_info=response.escalation_info,
            processing_time_ms=response.processing_time_ms,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            metadata=response.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Message processing failed: {str(e)}")


@router.post("/conversations/start", response_model=MessageResponse)
async def start_conversation(
    request: ConversationStartRequest,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """Start a new conversation with greeting message"""
    try:
        chat_request = ChatRequest(
            message="",  # Empty message for greeting
            client_id=request.client_id,
            session_id=request.session_id,
            user_id=request.user_id,
            metadata=request.metadata or {}
        )
        
        response = await chat_service.start_conversation(chat_request)
        
        return MessageResponse(
            message=response.message,
            conversation_id=response.conversation_id,
            response_type=response.response_type,
            confidence_score=response.confidence_score,
            should_escalate=response.should_escalate,
            escalation_info=response.escalation_info,
            processing_time_ms=response.processing_time_ms,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            metadata=response.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation start failed: {str(e)}")


@router.post("/conversations/{conversation_id}/end", response_model=MessageResponse)
async def end_conversation(
    conversation_id: str,
    client_id: str,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """End conversation with goodbye message"""
    try:
        response = await chat_service.end_conversation(conversation_id, client_id)
        
        return MessageResponse(
            message=response.message,
            conversation_id=response.conversation_id,
            response_type=response.response_type,
            confidence_score=response.confidence_score,
            should_escalate=response.should_escalate,
            escalation_info=response.escalation_info,
            processing_time_ms=response.processing_time_ms,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            metadata=response.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation end failed: {str(e)}")


@router.get("/conversations/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    limit: Optional[int] = None,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """Get conversation history"""
    try:
        messages = await chat_service.get_conversation_history(conversation_id, limit)
        
        # Get conversation metadata (simplified for this example)
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            total_messages=len(messages),
            conversation_state="active",  # Would get from actual conversation
            created_at=datetime.now(),  # Would get from actual conversation
            updated_at=datetime.now()   # Would get from actual conversation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")


@router.post("/clients/{client_id}/config")
async def update_client_config(
    client_id: str,
    config_request: ClientConfigRequest,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """Update client configuration"""
    try:
        # Convert request to internal configuration
        tone_config = ClientToneConfig(
            primary_tone=config_request.primary_tone,
            secondary_tone=config_request.secondary_tone,
            persona_description=config_request.persona_description,
            brand_voice=config_request.brand_voice,
            custom_instructions=config_request.custom_instructions,
            prohibited_words=config_request.prohibited_words,
            preferred_phrases=config_request.preferred_phrases,
            escalation_triggers=config_request.escalation_triggers,
            max_response_length=config_request.max_response_length
        )
        
        client_config = ClientConfiguration(
            client_id=client_id,
            tone_config=tone_config,
            knowledge_base_ids=config_request.knowledge_base_ids
        )
        
        await chat_service.update_client_configuration(client_id, client_config)
        
        return {"message": f"Configuration updated for client {client_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update client config: {str(e)}")


@router.get("/clients/{client_id}/escalations", response_model=List[EscalationResponse])
async def get_client_escalations(
    client_id: str,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """Get active escalations for client"""
    try:
        escalations = await chat_service.get_active_escalations(client_id)
        
        return [
            EscalationResponse(
                escalation_id=escalation["escalation_id"],
                conversation_id=escalation["conversation_id"],
                reason=escalation["reason"],
                priority=escalation["priority"],
                confidence_score=escalation["confidence_score"],
                sentiment_score=escalation["sentiment_score"],
                created_at=datetime.fromisoformat(escalation["created_at"]),
                status="active"  # Simplified
            )
            for escalation in escalations
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get escalations: {str(e)}")


@router.post("/escalations/{escalation_id}/resolve")
async def resolve_escalation(
    escalation_id: str,
    resolution_notes: str,
    agent_id: Optional[str] = None,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """Resolve an escalation"""
    try:
        await chat_service.resolve_escalation(escalation_id, resolution_notes, agent_id)
        return {"message": f"Escalation {escalation_id} resolved"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve escalation: {str(e)}")


@router.get("/stats", response_model=ServiceStatsResponse)
async def get_service_stats(
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """Get comprehensive service statistics"""
    try:
        stats = chat_service.get_service_stats()
        
        return ServiceStatsResponse(
            conversation_stats=stats["conversation_stats"],
            escalation_stats=stats["escalation_stats"],
            llm_metrics=stats["llm_metrics"],
            active_clients=stats["active_clients"],
            timestamp=stats["timestamp"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service stats: {str(e)}")


@router.post("/message/stream")
async def send_message_stream(
    request: MessageRequest,
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """
    Send message with streaming response
    
    Returns Server-Sent Events (SSE) stream of response chunks.
    """
    try:
        # For streaming, we'll need to modify the chat service to support streaming
        # For now, return regular response as stream
        chat_request = ChatRequest(
            message=request.message,
            client_id=request.client_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            metadata=request.metadata or {}
        )
        
        async def generate():
            # Process the message
            response = await chat_service.process_chat_message(chat_request)
            
            # Stream the response in chunks
            words = response.message.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "content": word + (" " if i < len(words) - 1 else ""),
                    "done": False,
                    "conversation_id": response.conversation_id
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)  # Simulate streaming delay
            
            # Send final metadata
            final_data = {
                "content": "",
                "done": True,
                "conversation_id": response.conversation_id,
                "metadata": {
                    "response_type": response.response_type,
                    "confidence_score": response.confidence_score,
                    "should_escalate": response.should_escalate,
                    "processing_time_ms": response.processing_time_ms,
                    "tokens_used": response.tokens_used,
                    "cost_usd": response.cost_usd
                }
            }
            yield f"data: {json.dumps(final_data)}\n\n"
        
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
        raise HTTPException(status_code=500, detail=f"Streaming message failed: {str(e)}")


@router.get("/health")
async def health_check(
    chat_service: ChatService = Depends(get_chat_service_dependency)
):
    """Health check for chat service"""
    try:
        # Basic health check
        stats = chat_service.get_service_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_conversations": stats["conversation_stats"]["active_conversations"],
            "active_escalations": stats["escalation_stats"]["total_active"],
            "llm_providers_available": len(stats["llm_metrics"]["available_providers"])
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }