"""
Client and tenant management models for multi-tenant architecture
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base

from app.core.database import Base

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
    """
    Client tenant model - represents a company/organization using the service
    """
    __tablename__ = "clients"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String(50), unique=True, nullable=False, index=True)  # Human-readable ID
    name = Column(String(200), nullable=False)
    
    # Contact information
    email = Column(String(255), nullable=False)
    contact_name = Column(String(100))
    phone = Column(String(20))
    website = Column(String(255))
    
    # Status and tier
    status = Column(String(20), nullable=False, default=ClientStatus.TRIAL)
    tier = Column(String(20), nullable=False, default=ClientTier.FREE)
    
    # Configuration
    configuration = Column(JSON, nullable=False, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    trial_ends_at = Column(DateTime(timezone=True))
    last_active_at = Column(DateTime(timezone=True))
    
    # Relationships
    api_keys = relationship("ClientAPIKey", back_populates="client", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", back_populates="client")
    conversations = relationship("Conversation", back_populates="client")
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
    
    def is_active(self) -> bool:
        """Check if client is active"""
        return self.status == ClientStatus.ACTIVE
    
    def is_trial_expired(self) -> bool:
        """Check if trial period has expired"""
        if not self.trial_ends_at:
            return False
        return datetime.now(timezone.utc) > self.trial_ends_at
    
    def update_last_active(self):
        """Update last active timestamp"""
        self.last_active_at = datetime.now(timezone.utc)

class ClientAPIKey(Base):
    """
    API keys for client authentication
    """
    __tablename__ = "client_api_keys"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    
    # Key information
    key_prefix = Column(String(20), nullable=False)  # First few characters for identification
    key_hash = Column(String(255), nullable=False)  # Hashed full key
    name = Column(String(100), nullable=False)  # Human-readable name
    description = Column(Text)
    
    # Permissions and limits
    scopes = Column(JSON, nullable=False, default=list)  # List of allowed scopes
    rate_limit = Column(Integer, default=1000)  # Requests per hour
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_used_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
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
    """
    Track client usage for billing and analytics
    """
    __tablename__ = "usage_records"
    
    # Primary identification
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
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    client = relationship("Client", back_populates="usage_records")
    
    # Indexes
    __table_args__ = (
        Index('idx_usage_client_period', 'client_id', 'period_start', 'period_end'),
        Index('idx_usage_period_start', 'period_start'),
    )

class ClientInvitation(Base):
    """
    Client invitations for onboarding
    """
    __tablename__ = "client_invitations"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Invitation details
    email = Column(String(255), nullable=False)
    client_name = Column(String(200), nullable=False)
    client_id = Column(String(50), nullable=False)
    invited_by = Column(String(255))  # Admin email who sent invitation
    
    # Invitation token
    token = Column(String(255), nullable=False, unique=True, index=True)
    
    # Status
    status = Column(String(20), nullable=False, default="pending")  # pending, accepted, expired
    expires_at = Column(DateTime(timezone=True), nullable=False)
    accepted_at = Column(DateTime(timezone=True))
    
    # Configuration for new client
    initial_configuration = Column(JSON, default=dict)
    initial_tier = Column(String(20), default=ClientTier.TRIAL)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_invitation_email_status', 'email', 'status'),
        Index('idx_invitation_expires_at', 'expires_at'),
    )
    
    def is_expired(self) -> bool:
        """Check if invitation is expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if invitation is valid for acceptance"""
        return self.status == "pending" and not self.is_expired()