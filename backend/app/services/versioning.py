from typing import List, Dict, Optional, Any, Tuple
import asyncio
import json
import hashlib
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from ..models.client import KnowledgeBase, Document, DocumentChunk, Client
from ..models.versioning import KnowledgeBaseVersion, DocumentVersion, VersionSnapshot
from ..services.vector_service import VectorService
from ..services.document_ingestion import DocumentIngestionService
from ..core.database import get_db


class VersionAction(Enum):
    """Types of version actions"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    ROLLBACK = "rollback"
    SNAPSHOT = "snapshot"


class VersionStatus(Enum):
    """Version status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class VersionChange:
    """Represents a change in a version"""
    action: VersionAction
    entity_type: str  # knowledge_base, document, chunk
    entity_id: int
    old_data: Optional[Dict[str, Any]] = None
    new_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VersionInfo:
    """Information about a version"""
    version_id: int
    version_number: str
    description: str
    created_at: datetime
    created_by: str
    status: VersionStatus
    changes: List[VersionChange]
    snapshot_size: int
    is_current: bool


class KnowledgeBaseVersioning:
    """Service for managing knowledge base versions"""
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_service = VectorService(db)
        self.document_service = DocumentIngestionService(db, auto_generate_embeddings=False)
    
    async def create_version(
        self,
        knowledge_base_id: int,
        description: str,
        created_by: str,
        changes: List[VersionChange] = None,
        auto_snapshot: bool = True
    ) -> KnowledgeBaseVersion:
        """Create a new version of a knowledge base"""
        
        # Get the knowledge base
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if not kb:
            raise ValueError(f"Knowledge base {knowledge_base_id} not found")
        
        # Get the current version number
        current_version = self._get_current_version(knowledge_base_id)
        new_version_number = self._increment_version_number(current_version.version if current_version else "0.0.0")
        
        # Create version record
        version = KnowledgeBaseVersion(
            knowledge_base_id=knowledge_base_id,
            version=new_version_number,
            description=description,
            created_by=created_by,
            status=VersionStatus.ACTIVE.value,
            changes=json.dumps([change.__dict__ for change in changes]) if changes else "[]",
            metadata={}
        )
        
        # Mark current version as inactive
        if current_version:
            current_version.status = VersionStatus.INACTIVE.value
            current_version.is_current = False
        
        version.is_current = True
        
        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)
        
        # Create snapshot if requested
        if auto_snapshot:
            await self._create_snapshot(version.id)
        
        return version
    
    async def rollback_to_version(
        self,
        knowledge_base_id: int,
        target_version_id: int,
        created_by: str,
        description: str = None
    ) -> KnowledgeBaseVersion:
        """Rollback knowledge base to a specific version"""
        
        # Get the target version
        target_version = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.id == target_version_id,
            KnowledgeBaseVersion.knowledge_base_id == knowledge_base_id
        ).first()
        
        if not target_version:
            raise ValueError(f"Version {target_version_id} not found")
        
        # Get the snapshot for the target version
        snapshot = self.db.query(VersionSnapshot).filter(
            VersionSnapshot.version_id == target_version_id
        ).first()
        
        if not snapshot:
            raise ValueError(f"No snapshot found for version {target_version_id}")
        
        # Restore from snapshot
        await self._restore_from_snapshot(snapshot, knowledge_base_id)
        
        # Create new version for the rollback
        rollback_description = description or f"Rollback to version {target_version.version}"
        changes = [VersionChange(
            action=VersionAction.ROLLBACK,
            entity_type="knowledge_base",
            entity_id=knowledge_base_id,
            metadata={"target_version_id": target_version_id, "target_version": target_version.version}
        )]
        
        new_version = await self.create_version(
            knowledge_base_id=knowledge_base_id,
            description=rollback_description,
            created_by=created_by,
            changes=changes
        )
        
        return new_version
    
    def get_version_history(
        self,
        knowledge_base_id: int,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False
    ) -> List[VersionInfo]:
        """Get version history for a knowledge base"""
        
        query = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.knowledge_base_id == knowledge_base_id
        )
        
        if not include_archived:
            query = query.filter(KnowledgeBaseVersion.status != VersionStatus.ARCHIVED.value)
        
        versions = query.order_by(desc(KnowledgeBaseVersion.created_at)).offset(offset).limit(limit).all()
        
        version_infos = []
        for version in versions:
            # Parse changes
            changes = []
            try:
                change_data = json.loads(version.changes)
                for change_dict in change_data:
                    changes.append(VersionChange(**change_dict))
            except (json.JSONDecodeError, TypeError):
                changes = []
            
            # Get snapshot size
            snapshot = self.db.query(VersionSnapshot).filter(
                VersionSnapshot.version_id == version.id
            ).first()
            snapshot_size = len(snapshot.snapshot_data) if snapshot else 0
            
            version_info = VersionInfo(
                version_id=version.id,
                version_number=version.version,
                description=version.description,
                created_at=version.created_at,
                created_by=version.created_by,
                status=VersionStatus(version.status),
                changes=changes,
                snapshot_size=snapshot_size,
                is_current=version.is_current
            )
            version_infos.append(version_info)
        
        return version_infos
    
    def get_current_version(self, knowledge_base_id: int) -> Optional[KnowledgeBaseVersion]:
        """Get the current active version of a knowledge base"""
        return self._get_current_version(knowledge_base_id)
    
    def compare_versions(
        self,
        knowledge_base_id: int,
        version1_id: int,
        version2_id: int
    ) -> Dict[str, Any]:
        """Compare two versions of a knowledge base"""
        
        # Get both versions
        version1 = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.id == version1_id,
            KnowledgeBaseVersion.knowledge_base_id == knowledge_base_id
        ).first()
        
        version2 = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.id == version2_id,
            KnowledgeBaseVersion.knowledge_base_id == knowledge_base_id
        ).first()
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        # Get snapshots
        snapshot1 = self.db.query(VersionSnapshot).filter(
            VersionSnapshot.version_id == version1_id
        ).first()
        
        snapshot2 = self.db.query(VersionSnapshot).filter(
            VersionSnapshot.version_id == version2_id
        ).first()
        
        # Compare snapshots
        comparison = {
            "version1": {
                "id": version1.id,
                "version": version1.version,
                "description": version1.description,
                "created_at": version1.created_at.isoformat(),
                "created_by": version1.created_by
            },
            "version2": {
                "id": version2.id,
                "version": version2.version,
                "description": version2.description,
                "created_at": version2.created_at.isoformat(),
                "created_by": version2.created_by
            },
            "differences": self._compute_differences(snapshot1, snapshot2)
        }
        
        return comparison
    
    async def archive_version(self, version_id: int, reason: str = None) -> bool:
        """Archive a version (soft delete)"""
        
        version = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.id == version_id
        ).first()
        
        if not version:
            return False
        
        if version.is_current:
            raise ValueError("Cannot archive the current version")
        
        version.status = VersionStatus.ARCHIVED.value
        version.metadata = version.metadata or {}
        version.metadata["archived_reason"] = reason
        version.metadata["archived_at"] = datetime.now(timezone.utc).isoformat()
        
        self.db.commit()
        return True
    
    async def delete_version(self, version_id: int, permanent: bool = False) -> bool:
        """Delete a version (soft or hard delete)"""
        
        version = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.id == version_id
        ).first()
        
        if not version:
            return False
        
        if version.is_current:
            raise ValueError("Cannot delete the current version")
        
        if permanent:
            # Hard delete - remove from database
            # Also delete associated snapshot
            snapshot = self.db.query(VersionSnapshot).filter(
                VersionSnapshot.version_id == version_id
            ).first()
            if snapshot:
                self.db.delete(snapshot)
            
            self.db.delete(version)
        else:
            # Soft delete - mark as deleted
            version.status = VersionStatus.DELETED.value
            version.metadata = version.metadata or {}
            version.metadata["deleted_at"] = datetime.now(timezone.utc).isoformat()
        
        self.db.commit()
        return True
    
    def get_version_statistics(self, knowledge_base_id: int) -> Dict[str, Any]:
        """Get statistics about versions for a knowledge base"""
        
        # Get all versions
        versions = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.knowledge_base_id == knowledge_base_id
        ).all()
        
        if not versions:
            return {
                "total_versions": 0,
                "current_version": None,
                "oldest_version": None,
                "newest_version": None,
                "status_distribution": {},
                "total_snapshots": 0,
                "total_snapshot_size": 0
            }
        
        # Calculate statistics
        current_version = next((v for v in versions if v.is_current), None)
        oldest_version = min(versions, key=lambda v: v.created_at)
        newest_version = max(versions, key=lambda v: v.created_at)
        
        status_distribution = {}
        for status in VersionStatus:
            count = len([v for v in versions if v.status == status.value])
            if count > 0:
                status_distribution[status.value] = count
        
        # Get snapshot information
        snapshots = self.db.query(VersionSnapshot).filter(
            VersionSnapshot.version_id.in_([v.id for v in versions])
        ).all()
        
        total_snapshot_size = sum(len(s.snapshot_data) for s in snapshots)
        
        return {
            "total_versions": len(versions),
            "current_version": {
                "id": current_version.id,
                "version": current_version.version,
                "description": current_version.description,
                "created_at": current_version.created_at.isoformat()
            } if current_version else None,
            "oldest_version": {
                "id": oldest_version.id,
                "version": oldest_version.version,
                "created_at": oldest_version.created_at.isoformat()
            },
            "newest_version": {
                "id": newest_version.id,
                "version": newest_version.version,
                "created_at": newest_version.created_at.isoformat()
            },
            "status_distribution": status_distribution,
            "total_snapshots": len(snapshots),
            "total_snapshot_size": total_snapshot_size
        }
    
    async def create_snapshot_for_current_version(self, knowledge_base_id: int) -> bool:
        """Create a snapshot for the current version"""
        
        current_version = self._get_current_version(knowledge_base_id)
        if not current_version:
            return False
        
        return await self._create_snapshot(current_version.id)
    
    # Private methods
    
    def _get_current_version(self, knowledge_base_id: int) -> Optional[KnowledgeBaseVersion]:
        """Get the current active version"""
        return self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.knowledge_base_id == knowledge_base_id,
            KnowledgeBaseVersion.is_current == True
        ).first()
    
    def _increment_version_number(self, current_version: str) -> str:
        """Increment version number (semantic versioning)"""
        try:
            major, minor, patch = map(int, current_version.split('.'))
            # For now, increment minor version
            return f"{major}.{minor + 1}.0"
        except ValueError:
            # If version format is invalid, start over
            return "1.0.0"
    
    async def _create_snapshot(self, version_id: int) -> bool:
        """Create a snapshot of the current state"""
        
        version = self.db.query(KnowledgeBaseVersion).filter(
            KnowledgeBaseVersion.id == version_id
        ).first()
        
        if not version:
            return False
        
        # Check if snapshot already exists
        existing_snapshot = self.db.query(VersionSnapshot).filter(
            VersionSnapshot.version_id == version_id
        ).first()
        
        if existing_snapshot:
            return True  # Snapshot already exists
        
        # Get all documents and chunks for this knowledge base
        documents = self.db.query(Document).filter(
            Document.knowledge_base_id == version.knowledge_base_id
        ).all()
        
        # Create snapshot data
        snapshot_data = {
            "knowledge_base": {
                "id": version.knowledge_base_id,
                "name": version.knowledge_base.name,
                "description": version.knowledge_base.description,
                "chunking_config": version.knowledge_base.chunking_config,
                "embedding_config": version.knowledge_base.embedding_config
            },
            "documents": [],
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_documents": len(documents),
                "total_chunks": 0
            }
        }
        
        total_chunks = 0
        for doc in documents:
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).all()
            
            doc_data = {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "content_type": doc.content_type,
                "source_url": doc.source_url,
                "metadata": doc.metadata,
                "processing_status": doc.processing_status,
                "content_hash": doc.content_hash,
                "version": doc.version,
                "chunks": []
            }
            
            for chunk in chunks:
                chunk_data = {
                    "id": chunk.id,
                    "chunk_text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "vector_id": chunk.vector_id
                }
                doc_data["chunks"].append(chunk_data)
                total_chunks += 1
            
            snapshot_data["documents"].append(doc_data)
        
        snapshot_data["metadata"]["total_chunks"] = total_chunks
        
        # Create snapshot record
        snapshot = VersionSnapshot(
            version_id=version_id,
            snapshot_data=json.dumps(snapshot_data),
            created_at=datetime.now(timezone.utc)
        )
        
        self.db.add(snapshot)
        self.db.commit()
        
        return True
    
    async def _restore_from_snapshot(self, snapshot: VersionSnapshot, knowledge_base_id: int):
        """Restore knowledge base from snapshot"""
        
        try:
            snapshot_data = json.loads(snapshot.snapshot_data)
        except json.JSONDecodeError:
            raise ValueError("Invalid snapshot data")
        
        # Clear current documents and chunks
        current_documents = self.db.query(Document).filter(
            Document.knowledge_base_id == knowledge_base_id
        ).all()
        
        for doc in current_documents:
            # Delete embeddings
            if self.vector_service:
                try:
                    await self.vector_service.delete_document_embeddings(doc.id)
                except Exception as e:
                    print(f"Warning: Failed to delete embeddings for document {doc.id}: {e}")
            
            # Delete chunks
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).delete()
            
            # Delete document
            self.db.delete(doc)
        
        # Update knowledge base configuration
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if kb and "knowledge_base" in snapshot_data:
            kb_data = snapshot_data["knowledge_base"]
            kb.name = kb_data.get("name", kb.name)
            kb.description = kb_data.get("description", kb.description)
            kb.chunking_config = kb_data.get("chunking_config", kb.chunking_config)
            kb.embedding_config = kb_data.get("embedding_config", kb.embedding_config)
        
        # Restore documents and chunks
        if "documents" in snapshot_data:
            for doc_data in snapshot_data["documents"]:
                # Create document
                doc = Document(
                    knowledge_base_id=knowledge_base_id,
                    title=doc_data["title"],
                    content=doc_data["content"],
                    content_type=doc_data["content_type"],
                    source_url=doc_data.get("source_url"),
                    metadata=doc_data.get("metadata", {}),
                    processing_status=doc_data.get("processing_status", "completed"),
                    content_hash=doc_data.get("content_hash"),
                    version=doc_data.get("version", "1.0.0")
                )
                
                self.db.add(doc)
                self.db.flush()  # Get the document ID
                
                # Create chunks
                for chunk_data in doc_data.get("chunks", []):
                    chunk = DocumentChunk(
                        document_id=doc.id,
                        chunk_text=chunk_data["chunk_text"],
                        chunk_index=chunk_data["chunk_index"],
                        metadata=chunk_data.get("metadata", {}),
                        vector_id=chunk_data.get("vector_id")
                    )
                    self.db.add(chunk)
        
        self.db.commit()
        
        # Regenerate embeddings for restored content
        if self.vector_service:
            try:
                await self.vector_service.generate_and_store_embeddings(knowledge_base_id)
            except Exception as e:
                print(f"Warning: Failed to regenerate embeddings: {e}")
    
    def _compute_differences(self, snapshot1: VersionSnapshot, snapshot2: VersionSnapshot) -> Dict[str, Any]:
        """Compute differences between two snapshots"""
        
        if not snapshot1 or not snapshot2:
            return {"error": "One or both snapshots not found"}
        
        try:
            data1 = json.loads(snapshot1.snapshot_data)
            data2 = json.loads(snapshot2.snapshot_data)
        except json.JSONDecodeError:
            return {"error": "Invalid snapshot data"}
        
        differences = {
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
        
        # Create document maps by ID
        docs1 = {doc["id"]: doc for doc in data1.get("documents", [])}
        docs2 = {doc["id"]: doc for doc in data2.get("documents", [])}
        
        # Find added documents (in data2 but not in data1)
        for doc_id, doc in docs2.items():
            if doc_id not in docs1:
                differences["documents"]["added"].append({
                    "id": doc_id,
                    "title": doc["title"],
                    "content_type": doc["content_type"]
                })
        
        # Find removed documents (in data1 but not in data2)
        for doc_id, doc in docs1.items():
            if doc_id not in docs2:
                differences["documents"]["removed"].append({
                    "id": doc_id,
                    "title": doc["title"],
                    "content_type": doc["content_type"]
                })
        
        # Find modified documents (in both but different)
        for doc_id in docs1.keys() & docs2.keys():
            doc1 = docs1[doc_id]
            doc2 = docs2[doc_id]
            
            changes = []
            
            # Compare basic fields
            fields_to_compare = ["title", "content", "content_type", "source_url", "metadata"]
            for field in fields_to_compare:
                if doc1.get(field) != doc2.get(field):
                    changes.append({
                        "field": field,
                        "old_value": doc1.get(field),
                        "new_value": doc2.get(field)
                    })
            
            # Compare chunks
            chunks1 = {c["id"]: c for c in doc1.get("chunks", [])}
            chunks2 = {c["id"]: c for c in doc2.get("chunks", [])}
            
            chunks_added = len(chunks2) - len(chunks1 & chunks2)
            chunks_removed = len(chunks1) - len(chunks1 & chunks2)
            
            if chunks_added > 0 or chunks_removed > 0:
                changes.append({
                    "field": "chunks",
                    "chunks_added": chunks_added,
                    "chunks_removed": chunks_removed
                })
            
            if changes:
                differences["documents"]["modified"].append({
                    "id": doc_id,
                    "title": doc1["title"],
                    "changes": changes
                })
        
        # Update summary
        differences["summary"]["documents_added"] = len(differences["documents"]["added"])
        differences["summary"]["documents_removed"] = len(differences["documents"]["removed"])
        differences["summary"]["documents_modified"] = len(differences["documents"]["modified"])
        differences["summary"]["total_changes"] = (
            differences["summary"]["documents_added"] +
            differences["summary"]["documents_removed"] +
            differences["summary"]["documents_modified"]
        )
        
        return differences


# Utility functions

def get_versioning_service(db: Session = None) -> KnowledgeBaseVersioning:
    """Get versioning service instance"""
    if db is None:
        db = next(get_db())
    return KnowledgeBaseVersioning(db)


async def auto_version_on_changes(
    knowledge_base_id: int,
    changes: List[VersionChange],
    created_by: str,
    description: str = None,
    db: Session = None
) -> KnowledgeBaseVersion:
    """Automatically create version when changes are detected"""
    if db is None:
        db = next(get_db())
    
    versioning = KnowledgeBaseVersioning(db)
    
    # Generate description if not provided
    if not description:
        change_types = list(set(change.action.value for change in changes))
        description = f"Auto-version: {', '.join(change_types)}"
    
    return await versioning.create_version(
        knowledge_base_id=knowledge_base_id,
        description=description,
        created_by=created_by,
        changes=changes
    )


def create_change_for_document_update(document_id: int, old_data: Dict, new_data: Dict) -> VersionChange:
    """Create a VersionChange for document update"""
    return VersionChange(
        action=VersionAction.UPDATE,
        entity_type="document",
        entity_id=document_id,
        old_data=old_data,
        new_data=new_data
    )


def create_change_for_document_creation(document_id: int, document_data: Dict) -> VersionChange:
    """Create a VersionChange for document creation"""
    return VersionChange(
        action=VersionAction.CREATE,
        entity_type="document",
        entity_id=document_id,
        new_data=document_data
    )


def create_change_for_document_deletion(document_id: int, document_data: Dict) -> VersionChange:
    """Create a VersionChange for document deletion"""
    return VersionChange(
        action=VersionAction.DELETE,
        entity_type="document",
        entity_id=document_id,
        old_data=document_data
    )