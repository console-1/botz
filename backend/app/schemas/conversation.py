from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ESCALATED = "escalated"


class ChatMessageRequest(BaseModel):
    """Request schema for sending a chat message"""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100)
    user_id: Optional[str] = Field(None, max_length=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessageResponse(BaseModel):
    """Response schema for chat message"""
    message_id: str
    conversation_id: str
    content: str
    role: MessageRole
    
    # Processing info
    processing_time_ms: int
    confidence_score: Optional[float]
    
    # RAG info
    retrieved_chunks: List[Dict[str, Any]]
    
    # Metadata
    metadata: Dict[str, Any]
    
    created_at: datetime


class ConversationCreateRequest(BaseModel):
    """Request schema for creating a new conversation"""
    client_id: str = Field(..., min_length=1, max_length=50)
    user_id: Optional[str] = Field(None, max_length=100)
    session_id: str = Field(..., min_length=1, max_length=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """Response schema for conversation data"""
    id: int
    conversation_id: str
    client_id: str
    user_id: Optional[str]
    session_id: str
    
    status: ConversationStatus
    is_escalated: bool
    escalation_reason: Optional[str]
    
    total_messages: int
    satisfaction_score: Optional[float]
    
    metadata: Dict[str, Any]
    
    created_at: datetime
    updated_at: Optional[datetime]
    last_message_at: datetime
    
    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Response schema for conversation list"""
    conversations: List[ConversationResponse]
    total: int
    page: int
    per_page: int


class MessageResponse(BaseModel):
    """Response schema for message data"""
    id: int
    message_id: str
    conversation_id: str
    content: str
    role: MessageRole
    
    processing_time_ms: Optional[int]
    token_count: Optional[int]
    cost_usd: Optional[float]
    
    model_used: Optional[str]
    provider: Optional[str]
    
    retrieved_chunks: List[Dict[str, Any]]
    confidence_score: Optional[float]
    
    metadata: Dict[str, Any]
    
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history"""
    conversation: ConversationResponse
    messages: List[MessageResponse]


class FeedbackRequest(BaseModel):
    """Request schema for conversation feedback"""
    conversation_id: str = Field(..., min_length=1, max_length=100)
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=1000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EscalationRequest(BaseModel):
    """Request schema for escalating a conversation"""
    conversation_id: str = Field(..., min_length=1, max_length=100)
    reason: str = Field(..., min_length=1, max_length=500)
    agent_notes: Optional[str] = Field(None, max_length=1000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsRequest(BaseModel):
    """Request schema for analytics queries"""
    client_id: str = Field(..., min_length=1, max_length=50)
    start_date: datetime
    end_date: datetime
    metrics: List[str] = Field(default_factory=lambda: ["messages", "conversations", "satisfaction"])
    group_by: Optional[str] = Field(None, regex="^(day|week|month)$")


class AnalyticsResponse(BaseModel):
    """Response schema for analytics data"""
    client_id: str
    period: Dict[str, datetime]
    metrics: Dict[str, Any]
    trends: Optional[Dict[str, Any]]