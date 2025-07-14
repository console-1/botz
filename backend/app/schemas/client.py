from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class VoiceType(str, Enum):
    PROFESSIONAL = "professional"
    WARM = "warm"
    CASUAL = "casual"
    TECHNICAL = "technical"
    PLAYFUL = "playful"


class PersonaType(str, Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EMPATHETIC = "empathetic"
    AUTHORITATIVE = "authoritative"
    HELPFUL = "helpful"


class ClientConfig(BaseModel):
    """Client configuration schema"""
    # Bot behavior
    voice: VoiceType = VoiceType.PROFESSIONAL
    persona: PersonaType = PersonaType.HELPFUL
    
    # Response settings
    max_response_length: int = Field(default=500, ge=100, le=2000)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Conversation settings
    max_context_turns: int = Field(default=10, ge=1, le=50)
    enable_clarifying_questions: bool = True
    enable_proactive_suggestions: bool = False
    
    # Fallback responses
    no_data_reply: str = "I don't have information about that specific topic. Let me connect you with a human agent who can help."
    escalation_message: str = "I'll connect you with a human agent who can better assist you with this request."
    error_message: str = "I'm experiencing some technical difficulties. Please try again in a moment."
    
    # Knowledge base settings
    search_limit: int = Field(default=5, ge=1, le=20)
    use_hybrid_search: bool = True
    rerank_results: bool = True


class ClientBranding(BaseModel):
    """Client branding configuration"""
    # Colors
    primary_color: str = "#007bff"
    secondary_color: str = "#6c757d"
    accent_color: str = "#28a745"
    text_color: str = "#212529"
    background_color: str = "#ffffff"
    
    # Logo and images
    logo_url: Optional[str] = None
    favicon_url: Optional[str] = None
    background_image_url: Optional[str] = None
    
    # Typography
    font_family: str = "system-ui, -apple-system, sans-serif"
    font_size: str = "14px"
    
    # Chat widget appearance
    widget_position: str = "bottom-right"  # bottom-right, bottom-left, etc.
    widget_size: str = "medium"  # small, medium, large
    chat_bubble_style: str = "rounded"  # rounded, square, pill
    
    # Custom CSS
    custom_css: Optional[str] = None


class ClientFeatures(BaseModel):
    """Client feature toggles"""
    # Core features
    file_uploads: bool = False
    voice_messages: bool = False
    emoji_reactions: bool = True
    typing_indicators: bool = True
    
    # Advanced features
    conversation_history: bool = True
    user_authentication: bool = False
    sentiment_analysis: bool = False
    conversation_rating: bool = True
    
    # Integration features
    webhook_notifications: bool = False
    crm_integration: bool = False
    analytics_tracking: bool = True
    
    # Business features
    business_hours_only: bool = False
    queue_management: bool = False
    agent_handoff: bool = True


class ClientCreateRequest(BaseModel):
    """Request schema for creating a new client"""
    client_id: str = Field(..., min_length=3, max_length=50, regex="^[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., regex="^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$")
    
    config: ClientConfig = ClientConfig()
    branding: ClientBranding = ClientBranding()
    features: ClientFeatures = ClientFeatures()


class ClientUpdateRequest(BaseModel):
    """Request schema for updating a client"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    email: Optional[str] = Field(None, regex="^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$")
    
    config: Optional[ClientConfig] = None
    branding: Optional[ClientBranding] = None
    features: Optional[ClientFeatures] = None
    
    is_active: Optional[bool] = None
    is_whitelabel: Optional[bool] = None


class ClientResponse(BaseModel):
    """Response schema for client data"""
    id: int
    client_id: str
    name: str
    email: str
    
    config: ClientConfig
    branding: ClientBranding
    features: ClientFeatures
    
    is_active: bool
    is_whitelabel: bool
    
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ClientListResponse(BaseModel):
    """Response schema for client list"""
    clients: List[ClientResponse]
    total: int
    page: int
    per_page: int


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base configuration"""
    # Chunking settings
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    chunking_strategy: str = "semantic"  # semantic, fixed, paragraph
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Processing settings
    auto_update: bool = True
    version_control: bool = True
    duplicate_detection: bool = True


class KnowledgeBaseCreateRequest(BaseModel):
    """Request schema for creating a knowledge base"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    
    config: KnowledgeBaseConfig = KnowledgeBaseConfig()


class KnowledgeBaseResponse(BaseModel):
    """Response schema for knowledge base data"""
    id: int
    client_id: str
    name: str
    description: Optional[str]
    
    version: str
    is_active: bool
    
    total_documents: int
    total_chunks: int
    last_updated: datetime
    
    config: KnowledgeBaseConfig
    
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True