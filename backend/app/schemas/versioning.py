from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class VersionActionEnum(str, Enum):
    """Version action types"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    ROLLBACK = "rollback"
    SNAPSHOT = "snapshot"


class VersionStatusEnum(str, Enum):
    """Version status types"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class VersionChange(BaseModel):
    """Schema for version change information"""
    action: VersionActionEnum
    entity_type: str = Field(..., description="Type of entity (knowledge_base, document, chunk)")
    entity_id: int
    old_data: Optional[Dict[str, Any]] = None
    new_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "action": "update",
                "entity_type": "document",
                "entity_id": 123,
                "old_data": {"title": "Old Title"},
                "new_data": {"title": "New Title"},
                "metadata": {"updated_by": "user123"}
            }
        }


class VersionCreateRequest(BaseModel):
    """Request schema for creating a new version"""
    description: str = Field(..., description="Description of the version")
    created_by: str = Field(..., description="User who created the version")
    changes: Optional[List[VersionChange]] = Field(None, description="List of changes in this version")
    auto_snapshot: bool = Field(True, description="Automatically create a snapshot")
    
    class Config:
        schema_extra = {
            "example": {
                "description": "Updated installation documentation",
                "created_by": "user123",
                "changes": [
                    {
                        "action": "update",
                        "entity_type": "document",
                        "entity_id": 123,
                        "old_data": {"title": "Old Title"},
                        "new_data": {"title": "New Title"}
                    }
                ],
                "auto_snapshot": True
            }
        }


class VersionResponse(BaseModel):
    """Response schema for version information"""
    id: int
    knowledge_base_id: int
    version: str
    description: str
    created_by: str
    status: str
    is_current: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    changes: str  # JSON string of changes
    metadata: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "knowledge_base_id": 10,
                "version": "1.2.0",
                "description": "Updated installation documentation",
                "created_by": "user123",
                "status": "active",
                "is_current": True,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "changes": "[{\"action\": \"update\", \"entity_type\": \"document\", \"entity_id\": 123}]",
                "metadata": {"snapshot_size": 1024}
            }
        }


class VersionHistoryResponse(BaseModel):
    """Response schema for version history"""
    knowledge_base_id: int
    versions: List[VersionResponse]
    total_versions: int
    limit: int
    offset: int
    has_more: bool
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 10,
                "versions": [
                    {
                        "id": 1,
                        "knowledge_base_id": 10,
                        "version": "1.2.0",
                        "description": "Updated installation documentation",
                        "created_by": "user123",
                        "status": "active",
                        "is_current": True,
                        "created_at": "2024-01-15T10:30:00Z",
                        "changes": "[]",
                        "metadata": {}
                    }
                ],
                "total_versions": 5,
                "limit": 50,
                "offset": 0,
                "has_more": False
            }
        }


class VersionRollbackRequest(BaseModel):
    """Request schema for version rollback"""
    created_by: str = Field(..., description="User performing the rollback")
    description: Optional[str] = Field(None, description="Description of the rollback")
    
    class Config:
        schema_extra = {
            "example": {
                "created_by": "user123",
                "description": "Rolling back due to issues with latest version"
            }
        }


class VersionDifference(BaseModel):
    """Schema for version differences"""
    documents: Dict[str, List[Dict[str, Any]]]
    summary: Dict[str, int]
    
    class Config:
        schema_extra = {
            "example": {
                "documents": {
                    "added": [{"id": 124, "title": "New Document"}],
                    "removed": [{"id": 125, "title": "Old Document"}],
                    "modified": [{"id": 126, "title": "Modified Document", "changes": []}]
                },
                "summary": {
                    "total_changes": 3,
                    "documents_added": 1,
                    "documents_removed": 1,
                    "documents_modified": 1
                }
            }
        }


class VersionInfo(BaseModel):
    """Schema for version information in comparisons"""
    id: int
    version: str
    description: str
    created_at: str
    created_by: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "version": "1.2.0",
                "description": "Updated installation documentation",
                "created_at": "2024-01-15T10:30:00Z",
                "created_by": "user123"
            }
        }


class VersionComparisonResponse(BaseModel):
    """Response schema for version comparison"""
    knowledge_base_id: int
    version1: VersionInfo
    version2: VersionInfo
    differences: VersionDifference
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 10,
                "version1": {
                    "id": 1,
                    "version": "1.1.0",
                    "description": "Previous version",
                    "created_at": "2024-01-14T10:30:00Z",
                    "created_by": "user123"
                },
                "version2": {
                    "id": 2,
                    "version": "1.2.0",
                    "description": "Current version",
                    "created_at": "2024-01-15T10:30:00Z",
                    "created_by": "user123"
                },
                "differences": {
                    "documents": {
                        "added": [],
                        "removed": [],
                        "modified": []
                    },
                    "summary": {
                        "total_changes": 0,
                        "documents_added": 0,
                        "documents_removed": 0,
                        "documents_modified": 0
                    }
                }
            }
        }


class VersionStatisticsResponse(BaseModel):
    """Response schema for version statistics"""
    knowledge_base_id: int
    total_versions: int
    current_version: Optional[Dict[str, Any]]
    oldest_version: Optional[Dict[str, Any]]
    newest_version: Optional[Dict[str, Any]]
    status_distribution: Dict[str, int]
    total_snapshots: int
    total_snapshot_size: int
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 10,
                "total_versions": 5,
                "current_version": {
                    "id": 5,
                    "version": "1.4.0",
                    "description": "Latest version",
                    "created_at": "2024-01-15T10:30:00Z"
                },
                "oldest_version": {
                    "id": 1,
                    "version": "1.0.0",
                    "created_at": "2024-01-01T10:30:00Z"
                },
                "newest_version": {
                    "id": 5,
                    "version": "1.4.0",
                    "created_at": "2024-01-15T10:30:00Z"
                },
                "status_distribution": {
                    "active": 1,
                    "inactive": 3,
                    "archived": 1
                },
                "total_snapshots": 4,
                "total_snapshot_size": 1048576
            }
        }


class VersionTagRequest(BaseModel):
    """Request schema for adding version tags"""
    tag_name: str = Field(..., description="Name of the tag")
    tag_value: Optional[str] = Field(None, description="Value of the tag")
    description: Optional[str] = Field(None, description="Description of the tag")
    color: Optional[str] = Field(None, description="Color for the tag (hex)")
    created_by: str = Field(..., description="User who created the tag")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "tag_name": "stable",
                "tag_value": "production",
                "description": "Stable production version",
                "color": "#28a745",
                "created_by": "user123",
                "metadata": {"environment": "production"}
            }
        }


class VersionCommentRequest(BaseModel):
    """Request schema for adding version comments"""
    comment_text: str = Field(..., description="Text of the comment")
    comment_type: str = Field("note", description="Type of comment (note, issue, approval, rejection)")
    created_by: str = Field(..., description="User who created the comment")
    mentioned_users: Optional[List[str]] = Field(None, description="Users mentioned in the comment")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('comment_type')
    def validate_comment_type(cls, v):
        valid_types = ['note', 'issue', 'approval', 'rejection']
        if v not in valid_types:
            raise ValueError(f'comment_type must be one of: {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "comment_text": "This version looks good for production deployment",
                "comment_type": "approval",
                "created_by": "user123",
                "mentioned_users": ["user456", "user789"],
                "metadata": {"approval_required": True}
            }
        }


class VersionApprovalRequest(BaseModel):
    """Request schema for version approval"""
    status: str = Field(..., description="Approval status (pending, approved, rejected)")
    approver: str = Field(..., description="User performing the approval")
    approval_comment: Optional[str] = Field(None, description="Comment for the approval")
    required_approvals: int = Field(1, description="Number of required approvals")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['pending', 'approved', 'rejected']
        if v not in valid_statuses:
            raise ValueError(f'status must be one of: {valid_statuses}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "status": "approved",
                "approver": "user123",
                "approval_comment": "Version approved for production deployment",
                "required_approvals": 2
            }
        }


class VersionMetricsResponse(BaseModel):
    """Response schema for version metrics"""
    version_id: int
    search_count: int
    document_views: int
    average_response_time: int
    user_satisfaction: int
    accuracy_score: int
    relevance_score: int
    indexing_time: int
    search_performance: int
    storage_size: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "version_id": 1,
                "search_count": 1250,
                "document_views": 3200,
                "average_response_time": 145,
                "user_satisfaction": 4,
                "accuracy_score": 85,
                "relevance_score": 82,
                "indexing_time": 2400,
                "search_performance": 120,
                "storage_size": 1048576,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T15:45:00Z"
            }
        }


class VersionTag(BaseModel):
    """Schema for version tag information"""
    id: int
    tag_name: str
    tag_value: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    created_by: str
    created_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "tag_name": "stable",
                "tag_value": "production",
                "description": "Stable production version",
                "color": "#28a745",
                "created_by": "user123",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class VersionComment(BaseModel):
    """Schema for version comment information"""
    id: int
    comment_text: str
    comment_type: str
    created_by: str
    mentioned_users: List[str]
    is_resolved: bool
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "comment_text": "This version looks good for production deployment",
                "comment_type": "approval",
                "created_by": "user123",
                "mentioned_users": ["user456"],
                "is_resolved": False,
                "resolved_by": None,
                "resolved_at": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": None
            }
        }


class VersionApproval(BaseModel):
    """Schema for version approval information"""
    id: int
    status: str
    approver: str
    approval_comment: Optional[str] = None
    required_approvals: int
    current_approvals: int
    created_at: datetime
    approved_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "status": "approved",
                "approver": "user123",
                "approval_comment": "Version approved for production deployment",
                "required_approvals": 2,
                "current_approvals": 2,
                "created_at": "2024-01-15T10:30:00Z",
                "approved_at": "2024-01-15T11:00:00Z"
            }
        }


class VersionSnapshot(BaseModel):
    """Schema for version snapshot information"""
    id: int
    version_id: int
    size_bytes: int
    document_count: int
    chunk_count: int
    compression_type: str
    checksum: Optional[str] = None
    created_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "version_id": 1,
                "size_bytes": 1048576,
                "document_count": 25,
                "chunk_count": 150,
                "compression_type": "gzip",
                "checksum": "sha256:abc123...",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


# Additional utility schemas

class VersionSummary(BaseModel):
    """Summary information for a version"""
    id: int
    version: str
    description: str
    is_current: bool
    created_at: datetime
    created_by: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "version": "1.2.0",
                "description": "Updated installation documentation",
                "is_current": True,
                "created_at": "2024-01-15T10:30:00Z",
                "created_by": "user123"
            }
        }


class VersionHealthResponse(BaseModel):
    """Response schema for versioning health check"""
    status: str
    total_versions: int
    total_snapshots: Any  # Can be int or string
    versioning_service: str
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "total_versions": 25,
                "total_snapshots": 20,
                "versioning_service": "operational",
                "error": None
            }
        }


class VersionOperationResponse(BaseModel):
    """Generic response for version operations"""
    message: str
    success: bool
    version_id: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Version archived successfully",
                "success": True,
                "version_id": 1,
                "details": {"archived_at": "2024-01-15T10:30:00Z"}
            }
        }