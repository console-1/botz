from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from ...core.database import get_db
from ...services.versioning import KnowledgeBaseVersioning, VersionAction, VersionChange
from ...models.client import Client, KnowledgeBase
from ...models.versioning import KnowledgeBaseVersion
from ...schemas.versioning import (
    VersionCreateRequest, VersionResponse, VersionHistoryResponse,
    VersionComparisonResponse, VersionStatisticsResponse, VersionRollbackRequest,
    VersionTagRequest, VersionCommentRequest, VersionApprovalRequest
)

router = APIRouter()


@router.post("/knowledge-bases/{kb_id}/versions", response_model=VersionResponse)
async def create_version(
    kb_id: int,
    version_request: VersionCreateRequest,
    db: Session = Depends(get_db)
):
    """Create a new version of a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Create versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    # Parse changes if provided
    changes = []
    if version_request.changes:
        for change_data in version_request.changes:
            change = VersionChange(
                action=VersionAction(change_data.action),
                entity_type=change_data.entity_type,
                entity_id=change_data.entity_id,
                old_data=change_data.old_data,
                new_data=change_data.new_data,
                metadata=change_data.metadata
            )
            changes.append(change)
    
    try:
        # Create version
        version = await versioning.create_version(
            knowledge_base_id=kb_id,
            description=version_request.description,
            created_by=version_request.created_by,
            changes=changes,
            auto_snapshot=version_request.auto_snapshot
        )
        
        return VersionResponse(
            id=version.id,
            knowledge_base_id=version.knowledge_base_id,
            version=version.version,
            description=version.description,
            created_by=version.created_by,
            status=version.status,
            is_current=version.is_current,
            created_at=version.created_at,
            updated_at=version.updated_at,
            changes=version.changes,
            metadata=version.metadata
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create version: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/versions", response_model=VersionHistoryResponse)
async def get_version_history(
    kb_id: int,
    limit: int = Query(50, ge=1, le=100, description="Number of versions to return"),
    offset: int = Query(0, ge=0, description="Number of versions to skip"),
    include_archived: bool = Query(False, description="Include archived versions"),
    db: Session = Depends(get_db)
):
    """Get version history for a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Get version history
        version_infos = versioning.get_version_history(
            knowledge_base_id=kb_id,
            limit=limit,
            offset=offset,
            include_archived=include_archived
        )
        
        # Convert to response format
        versions = []
        for info in version_infos:
            versions.append(VersionResponse(
                id=info.version_id,
                knowledge_base_id=kb_id,
                version=info.version_number,
                description=info.description,
                created_by=info.created_by,
                status=info.status.value,
                is_current=info.is_current,
                created_at=info.created_at,
                updated_at=info.created_at,  # Use created_at as fallback
                changes=json.dumps([change.__dict__ for change in info.changes]),
                metadata={"snapshot_size": info.snapshot_size}
            ))
        
        return VersionHistoryResponse(
            knowledge_base_id=kb_id,
            versions=versions,
            total_versions=len(versions),
            limit=limit,
            offset=offset,
            has_more=len(versions) == limit  # Simple check
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get version history: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/versions/current", response_model=VersionResponse)
async def get_current_version(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Get the current version of a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Get current version
        current_version = versioning.get_current_version(kb_id)
        
        if not current_version:
            raise HTTPException(status_code=404, detail="No current version found")
        
        return VersionResponse(
            id=current_version.id,
            knowledge_base_id=current_version.knowledge_base_id,
            version=current_version.version,
            description=current_version.description,
            created_by=current_version.created_by,
            status=current_version.status,
            is_current=current_version.is_current,
            created_at=current_version.created_at,
            updated_at=current_version.updated_at,
            changes=current_version.changes,
            metadata=current_version.metadata
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current version: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/versions/{version_id}/rollback", response_model=VersionResponse)
async def rollback_to_version(
    kb_id: int,
    version_id: int,
    rollback_request: VersionRollbackRequest,
    db: Session = Depends(get_db)
):
    """Rollback knowledge base to a specific version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Perform rollback
        new_version = await versioning.rollback_to_version(
            knowledge_base_id=kb_id,
            target_version_id=version_id,
            created_by=rollback_request.created_by,
            description=rollback_request.description
        )
        
        return VersionResponse(
            id=new_version.id,
            knowledge_base_id=new_version.knowledge_base_id,
            version=new_version.version,
            description=new_version.description,
            created_by=new_version.created_by,
            status=new_version.status,
            is_current=new_version.is_current,
            created_at=new_version.created_at,
            updated_at=new_version.updated_at,
            changes=new_version.changes,
            metadata=new_version.metadata
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rollback version: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/versions/{version1_id}/compare/{version2_id}", response_model=VersionComparisonResponse)
async def compare_versions(
    kb_id: int,
    version1_id: int,
    version2_id: int,
    db: Session = Depends(get_db)
):
    """Compare two versions of a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Compare versions
        comparison = versioning.compare_versions(
            knowledge_base_id=kb_id,
            version1_id=version1_id,
            version2_id=version2_id
        )
        
        return VersionComparisonResponse(
            knowledge_base_id=kb_id,
            version1=comparison["version1"],
            version2=comparison["version2"],
            differences=comparison["differences"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare versions: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/versions/statistics", response_model=VersionStatisticsResponse)
async def get_version_statistics(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Get statistics about versions for a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Get statistics
        stats = versioning.get_version_statistics(kb_id)
        
        return VersionStatisticsResponse(
            knowledge_base_id=kb_id,
            total_versions=stats["total_versions"],
            current_version=stats["current_version"],
            oldest_version=stats["oldest_version"],
            newest_version=stats["newest_version"],
            status_distribution=stats["status_distribution"],
            total_snapshots=stats["total_snapshots"],
            total_snapshot_size=stats["total_snapshot_size"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/versions/{version_id}/archive")
async def archive_version(
    kb_id: int,
    version_id: int,
    reason: Optional[str] = Query(None, description="Reason for archiving"),
    db: Session = Depends(get_db)
):
    """Archive a version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Archive version
        success = await versioning.archive_version(version_id, reason)
        
        if success:
            return {"message": f"Version {version_id} archived successfully"}
        else:
            raise HTTPException(status_code=404, detail="Version not found")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive version: {str(e)}")


@router.delete("/knowledge-bases/{kb_id}/versions/{version_id}")
async def delete_version(
    kb_id: int,
    version_id: int,
    permanent: bool = Query(False, description="Permanently delete (cannot be undone)"),
    db: Session = Depends(get_db)
):
    """Delete a version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Delete version
        success = await versioning.delete_version(version_id, permanent)
        
        if success:
            delete_type = "permanently deleted" if permanent else "deleted"
            return {"message": f"Version {version_id} {delete_type} successfully"}
        else:
            raise HTTPException(status_code=404, detail="Version not found")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete version: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/versions/{version_id}/snapshot")
async def create_snapshot(
    kb_id: int,
    version_id: int,
    db: Session = Depends(get_db)
):
    """Create a snapshot for a version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Create snapshot
        success = await versioning._create_snapshot(version_id)
        
        if success:
            return {"message": f"Snapshot created successfully for version {version_id}"}
        else:
            raise HTTPException(status_code=404, detail="Version not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/versions/current/snapshot")
async def create_current_snapshot(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Create a snapshot for the current version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get versioning service
    versioning = KnowledgeBaseVersioning(db)
    
    try:
        # Create snapshot for current version
        success = await versioning.create_snapshot_for_current_version(kb_id)
        
        if success:
            return {"message": "Snapshot created successfully for current version"}
        else:
            raise HTTPException(status_code=404, detail="No current version found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/versions/{version_id}/tags")
async def add_version_tag(
    kb_id: int,
    version_id: int,
    tag_request: VersionTagRequest,
    db: Session = Depends(get_db)
):
    """Add a tag to a version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Verify version exists
    version = db.query(KnowledgeBaseVersion).filter(
        KnowledgeBaseVersion.id == version_id,
        KnowledgeBaseVersion.knowledge_base_id == kb_id
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    try:
        from ...models.versioning import VersionTag
        
        # Create tag
        tag = VersionTag(
            version_id=version_id,
            tag_name=tag_request.tag_name,
            tag_value=tag_request.tag_value,
            description=tag_request.description,
            color=tag_request.color,
            created_by=tag_request.created_by,
            metadata=tag_request.metadata or {}
        )
        
        db.add(tag)
        db.commit()
        db.refresh(tag)
        
        return {
            "id": tag.id,
            "tag_name": tag.tag_name,
            "tag_value": tag.tag_value,
            "description": tag.description,
            "color": tag.color,
            "created_by": tag.created_by,
            "created_at": tag.created_at
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add tag: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/versions/{version_id}/tags")
async def get_version_tags(
    kb_id: int,
    version_id: int,
    db: Session = Depends(get_db)
):
    """Get all tags for a version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        from ...models.versioning import VersionTag
        
        # Get tags
        tags = db.query(VersionTag).filter(VersionTag.version_id == version_id).all()
        
        return {
            "version_id": version_id,
            "tags": [
                {
                    "id": tag.id,
                    "tag_name": tag.tag_name,
                    "tag_value": tag.tag_value,
                    "description": tag.description,
                    "color": tag.color,
                    "created_by": tag.created_by,
                    "created_at": tag.created_at
                }
                for tag in tags
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/versions/{version_id}/comments")
async def add_version_comment(
    kb_id: int,
    version_id: int,
    comment_request: VersionCommentRequest,
    db: Session = Depends(get_db)
):
    """Add a comment to a version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Verify version exists
    version = db.query(KnowledgeBaseVersion).filter(
        KnowledgeBaseVersion.id == version_id,
        KnowledgeBaseVersion.knowledge_base_id == kb_id
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    try:
        from ...models.versioning import VersionComment
        
        # Create comment
        comment = VersionComment(
            version_id=version_id,
            comment_text=comment_request.comment_text,
            comment_type=comment_request.comment_type,
            created_by=comment_request.created_by,
            mentioned_users=comment_request.mentioned_users or [],
            metadata=comment_request.metadata or {}
        )
        
        db.add(comment)
        db.commit()
        db.refresh(comment)
        
        return {
            "id": comment.id,
            "comment_text": comment.comment_text,
            "comment_type": comment.comment_type,
            "created_by": comment.created_by,
            "mentioned_users": comment.mentioned_users,
            "is_resolved": comment.is_resolved,
            "created_at": comment.created_at
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add comment: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/versions/{version_id}/comments")
async def get_version_comments(
    kb_id: int,
    version_id: int,
    db: Session = Depends(get_db)
):
    """Get all comments for a version"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        from ...models.versioning import VersionComment
        
        # Get comments
        comments = db.query(VersionComment).filter(
            VersionComment.version_id == version_id
        ).order_by(VersionComment.created_at.desc()).all()
        
        return {
            "version_id": version_id,
            "comments": [
                {
                    "id": comment.id,
                    "comment_text": comment.comment_text,
                    "comment_type": comment.comment_type,
                    "created_by": comment.created_by,
                    "mentioned_users": comment.mentioned_users,
                    "is_resolved": comment.is_resolved,
                    "resolved_by": comment.resolved_by,
                    "resolved_at": comment.resolved_at,
                    "created_at": comment.created_at,
                    "updated_at": comment.updated_at
                }
                for comment in comments
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get comments: {str(e)}")


@router.get("/versioning/health")
async def versioning_health(db: Session = Depends(get_db)):
    """Check health of versioning service"""
    
    try:
        # Check if we can create a versioning service
        versioning = KnowledgeBaseVersioning(db)
        
        # Get some basic stats
        total_versions = db.query(KnowledgeBaseVersion).count()
        total_snapshots = 0  # Simplified for now
        
        return {
            "status": "healthy",
            "total_versions": total_versions,
            "total_snapshots": total_snapshots,
            "versioning_service": "operational"
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Import json for responses
import json