from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base


class KnowledgeBaseVersion(Base):
    """Knowledge base version tracking"""
    __tablename__ = "knowledge_base_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    
    # Version information
    version = Column(String, nullable=False)  # e.g., "1.2.0"
    description = Column(Text, nullable=False)
    created_by = Column(String, nullable=False)
    
    # Status and flags
    status = Column(String, default="active")  # active, inactive, archived, deleted
    is_current = Column(Boolean, default=False)
    
    # Change tracking
    changes = Column(JSON, nullable=False, default=[])  # List of VersionChange objects
    metadata = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="versions")
    snapshots = relationship("VersionSnapshot", back_populates="version", cascade="all, delete-orphan")
    document_versions = relationship("DocumentVersion", back_populates="kb_version", cascade="all, delete-orphan")


class DocumentVersion(Base):
    """Document version tracking within knowledge base versions"""
    __tablename__ = "document_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    kb_version_id = Column(Integer, ForeignKey("knowledge_base_versions.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Version information
    document_version = Column(String, nullable=False)  # e.g., "1.0.0"
    action = Column(String, nullable=False)  # create, update, delete
    
    # Document state at this version
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String, default="text/plain")
    source_url = Column(String)
    metadata = Column(JSON, nullable=False, default={})
    content_hash = Column(String, index=True)
    
    # Change information
    changes = Column(JSON, nullable=False, default=[])  # Specific changes made
    diff_data = Column(JSON, nullable=False, default={})  # Diff from previous version
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    kb_version = relationship("KnowledgeBaseVersion", back_populates="document_versions")
    document = relationship("Document")


class VersionSnapshot(Base):
    """Complete snapshot of knowledge base state at a specific version"""
    __tablename__ = "version_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("knowledge_base_versions.id"), nullable=False, unique=True, index=True)
    
    # Snapshot data
    snapshot_data = Column(Text, nullable=False)  # JSON string of complete state
    compression_type = Column(String, default="none")  # none, gzip, lz4
    checksum = Column(String, index=True)  # For integrity verification
    
    # Metadata
    size_bytes = Column(Integer, default=0)
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    version = relationship("KnowledgeBaseVersion", back_populates="snapshots")


class VersionTag(Base):
    """Tags for versions (e.g., 'stable', 'beta', 'production')"""
    __tablename__ = "version_tags"
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("knowledge_base_versions.id"), nullable=False, index=True)
    
    # Tag information
    tag_name = Column(String, nullable=False)
    tag_value = Column(String)
    description = Column(Text)
    color = Column(String)  # Hex color for UI
    
    # Metadata
    created_by = Column(String, nullable=False)
    metadata = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    version = relationship("KnowledgeBaseVersion")


class VersionComment(Base):
    """Comments on versions for collaboration"""
    __tablename__ = "version_comments"
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("knowledge_base_versions.id"), nullable=False, index=True)
    
    # Comment information
    comment_text = Column(Text, nullable=False)
    comment_type = Column(String, default="note")  # note, issue, approval, rejection
    
    # User information
    created_by = Column(String, nullable=False)
    mentioned_users = Column(JSON, nullable=False, default=[])
    
    # Status
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(String)
    resolved_at = Column(DateTime(timezone=True))
    
    # Metadata
    metadata = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    version = relationship("KnowledgeBaseVersion")


class VersionApproval(Base):
    """Approval workflow for versions"""
    __tablename__ = "version_approvals"
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("knowledge_base_versions.id"), nullable=False, index=True)
    
    # Approval information
    status = Column(String, default="pending")  # pending, approved, rejected
    approver = Column(String, nullable=False)
    approval_comment = Column(Text)
    
    # Requirements
    required_approvals = Column(Integer, default=1)
    current_approvals = Column(Integer, default=0)
    
    # Metadata
    metadata = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    approved_at = Column(DateTime(timezone=True))
    
    # Relationships
    version = relationship("KnowledgeBaseVersion")


class VersionMetrics(Base):
    """Metrics and analytics for versions"""
    __tablename__ = "version_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("knowledge_base_versions.id"), nullable=False, index=True)
    
    # Usage metrics
    search_count = Column(Integer, default=0)
    document_views = Column(Integer, default=0)
    average_response_time = Column(Integer, default=0)  # milliseconds
    
    # Quality metrics
    user_satisfaction = Column(Integer, default=0)  # 1-5 scale
    accuracy_score = Column(Integer, default=0)  # 1-100 scale
    relevance_score = Column(Integer, default=0)  # 1-100 scale
    
    # Performance metrics
    indexing_time = Column(Integer, default=0)  # milliseconds
    search_performance = Column(Integer, default=0)  # milliseconds
    storage_size = Column(Integer, default=0)  # bytes
    
    # Metadata
    metadata = Column(JSON, nullable=False, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    version = relationship("KnowledgeBaseVersion")


# Add relationship back to KnowledgeBase
from .client import KnowledgeBase
KnowledgeBase.versions = relationship("KnowledgeBaseVersion", back_populates="knowledge_base", cascade="all, delete-orphan")