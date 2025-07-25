from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from enum import Enum
import uuid

from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates
from ..core.database import Base

class ClientStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TRIAL = "trial"

class ClientTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class Client(Base):
    __tablename__ = "clients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    client_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    email = Column(String(255), nullable=False)
    
    # Contact information
    contact_name = Column(String(100))
    phone = Column(String(20))
    website = Column(String(255))
    
    # Status and tier
    status = Column(String(20), nullable=False, default=ClientStatus.TRIAL)
    tier = Column(String(20), nullable=False, default=ClientTier.FREE)
    
    # Legacy compatibility
    is_active = Column(Boolean, default=True)
    is_whitelabel = Column(Boolean, default=False)
    
    # Configuration (merged legacy fields)
    config = Column(JSON, nullable=False, default={})
    branding = Column(JSON, nullable=False, default={})
    features = Column(JSON, nullable=False, default={})
    configuration = Column(JSON, nullable=False, default={})  # New unified config
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    trial_ends_at = Column(DateTime(timezone=True))
    last_active_at = Column(DateTime(timezone=True))
    
    # Relationships
    knowledge_bases = relationship("KnowledgeBase", back_populates="client")
    conversations = relationship("Conversation", back_populates="client")
    usage_metrics = relationship("UsageMetric", back_populates="client")
    api_keys = relationship("ClientAPIKey", back_populates="client", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="client")
    
    # Indexes
    __table_args__ = (
        Index('idx_client_status_tier', 'status', 'tier'),
        Index('idx_client_created_at', 'created_at'),
    )
    
    @validates('client_id')
    def validate_client_id(self, key, client_id):
        """Validate client_id format"""
        if not client_id or not client_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError("client_id must contain only alphanumeric characters, hyphens, and underscores")
        return client_id.lower()
    
    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for new clients"""
        return {
            "branding": {
                "company_name": self.name,
                "primary_color": "#007bff",
                "secondary_color": "#6c757d",
                "logo_url": None,
                "welcome_message": f"Hi! Welcome to {self.name}. How can we help you today?",
                "bot_name": "Assistant",
                "powered_by_text": "Powered by AI"
            },
            "behavior": {
                "max_conversation_length": 50,
                "response_timeout": 30,
                "escalation_threshold": 0.7,
                "enable_feedback": True,
                "enable_transcript_email": False
            },
            "security": {
                "allowed_domains": [],
                "enable_cors": True,
                "require_user_identification": False
            },
            "limits": {
                "messages_per_day": 1000,
                "conversations_per_day": 100,
                "api_calls_per_hour": 500,
                "knowledge_base_size_mb": 100
            }
        }
    
    def is_active_client(self) -> bool:
        """Check if client is active"""
        return self.status == ClientStatus.ACTIVE or self.is_active
    
    def is_trial_expired(self) -> bool:
        """Check if trial period has expired"""
        if not self.trial_ends_at:
            return False
        return datetime.now(timezone.utc) > self.trial_ends_at
    
    def update_last_active(self):
        """Update last active timestamp"""
        self.last_active_at = datetime.now(timezone.utc)

class ClientAPIKey(Base):
    """API keys for client authentication"""
    __tablename__ = "client_api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    
    # Key information
    key_prefix = Column(String(20), nullable=False)
    key_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Permissions and limits
    scopes = Column(JSON, nullable=False, default=list)
    rate_limit = Column(Integer, default=1000)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_used_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    client = relationship("Client", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_key_prefix', 'key_prefix'),
        Index('idx_api_key_client_active', 'client_id', 'is_active'),
    )
    
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid for use"""
        return self.is_active and not self.is_expired()
    
    def update_last_used(self):
        """Update last used timestamp"""
        self.last_used_at = datetime.now(timezone.utc)

class UsageRecord(Base):
    """Track client usage for billing and analytics"""
    __tablename__ = "usage_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    
    # Usage metrics
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Conversation metrics
    total_conversations = Column(Integer, nullable=False, default=0)
    total_messages = Column(Integer, nullable=False, default=0)
    total_tokens_used = Column(Integer, nullable=False, default=0)
    
    # API metrics
    api_calls_made = Column(Integer, nullable=False, default=0)
    successful_responses = Column(Integer, nullable=False, default=0)
    failed_responses = Column(Integer, nullable=False, default=0)
    
    # Performance metrics
    avg_response_time_ms = Column(Integer, default=0)
    escalations_triggered = Column(Integer, nullable=False, default=0)
    
    # Cost metrics
    estimated_cost_usd = Column(Integer, default=0)  # Store as cents
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    client = relationship("Client", back_populates="usage_records")
    
    # Indexes
    __table_args__ = (
        Index('idx_usage_client_period', 'client_id', 'period_start', 'period_end'),
        Index('idx_usage_period_start', 'period_start'),
    )


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False, index=True)
    client_string_id = Column(String, ForeignKey("clients.client_id"), nullable=True, index=True)  # Legacy support
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Versioning
    version = Column(String, default="1.0.0")
    is_active = Column(Boolean, default=True)
    
    # Metadata
    total_documents = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    
    # Configuration
    chunking_config = Column(JSON, nullable=False, default={})
    embedding_config = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    client = relationship("Client", back_populates="knowledge_bases")
    documents = relationship("Document", back_populates="knowledge_base")


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    
    # Document info
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String, default="text/plain")
    source_url = Column(String)
    
    # Metadata
    metadata = Column(JSON, nullable=False, default={})
    
    # Processing status
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    error_message = Column(Text)
    
    # Versioning
    version = Column(String, default="1.0.0")
    content_hash = Column(String, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Chunk info
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Metadata
    metadata = Column(JSON, nullable=False, default={})
    
    # Vector info (stored in Qdrant, referenced here)
    vector_id = Column(String, unique=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")