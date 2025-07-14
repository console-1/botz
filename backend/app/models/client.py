from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    
    # Configuration
    config = Column(JSON, nullable=False, default={})
    
    # Branding
    branding = Column(JSON, nullable=False, default={})
    
    # Features
    features = Column(JSON, nullable=False, default={})
    
    # Status
    is_active = Column(Boolean, default=True)
    is_whitelabel = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    knowledge_bases = relationship("KnowledgeBase", back_populates="client")
    conversations = relationship("Conversation", back_populates="client")
    usage_metrics = relationship("UsageMetric", back_populates="client")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False, index=True)
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