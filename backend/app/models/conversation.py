from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, unique=True, index=True, nullable=False)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False, index=True)
    client_string_id = Column(String, ForeignKey("clients.client_id"), nullable=True, index=True)  # Legacy support
    
    # User info
    user_id = Column(String, index=True)  # Optional, for authenticated users
    session_id = Column(String, nullable=False, index=True)
    
    # Conversation metadata
    metadata = Column(JSON, nullable=False, default={})
    
    # Status
    status = Column(String, default="active")  # active, closed, escalated
    is_escalated = Column(Boolean, default=False)
    escalation_reason = Column(Text)
    
    # Analytics
    total_messages = Column(Integer, default=0)
    satisfaction_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    client = relationship("Client", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"), nullable=False, index=True)
    
    # Message info
    message_id = Column(String, unique=True, index=True, nullable=False)
    content = Column(Text, nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    
    # Processing info
    processing_time_ms = Column(Integer)
    token_count = Column(Integer)
    cost_usd = Column(Float)
    
    # LLM info
    model_used = Column(String)
    provider = Column(String)
    
    # RAG info
    retrieved_chunks = Column(JSON, nullable=False, default=[])
    confidence_score = Column(Float)
    
    # Metadata
    metadata = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class UsageMetric(Base):
    __tablename__ = "usage_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False, index=True)
    client_string_id = Column(String, ForeignKey("clients.client_id"), nullable=True, index=True)  # Legacy support
    
    # Metrics
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    total_messages = Column(Integer, default=0)
    total_conversations = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    
    # Performance metrics
    avg_response_time_ms = Column(Float)
    avg_confidence_score = Column(Float)
    escalation_rate = Column(Float)
    satisfaction_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    client = relationship("Client", back_populates="usage_metrics")