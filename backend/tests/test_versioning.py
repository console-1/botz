import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.services.versioning import (
    KnowledgeBaseVersioning, VersionAction, VersionStatus, 
    VersionChange, VersionInfo, auto_version_on_changes
)
from app.models.client import KnowledgeBase, Document, DocumentChunk
from app.models.versioning import KnowledgeBaseVersion, VersionSnapshot


class TestVersionChange:
    """Test suite for VersionChange dataclass"""
    
    def test_version_change_creation(self):
        """Test VersionChange creation"""
        change = VersionChange(
            action=VersionAction.UPDATE,
            entity_type="document",
            entity_id=123,
            old_data={"title": "Old Title"},
            new_data={"title": "New Title"},
            metadata={"user": "test_user"}
        )
        
        assert change.action == VersionAction.UPDATE
        assert change.entity_type == "document"
        assert change.entity_id == 123
        assert change.old_data["title"] == "Old Title"
        assert change.new_data["title"] == "New Title"
        assert change.metadata["user"] == "test_user"
    
    def test_version_change_minimal(self):
        """Test VersionChange with minimal data"""
        change = VersionChange(
            action=VersionAction.CREATE,
            entity_type="document",
            entity_id=456
        )
        
        assert change.action == VersionAction.CREATE
        assert change.entity_type == "document"
        assert change.entity_id == 456
        assert change.old_data is None
        assert change.new_data is None
        assert change.metadata is None


class TestVersionInfo:
    """Test suite for VersionInfo dataclass"""
    
    def test_version_info_creation(self):
        """Test VersionInfo creation"""
        changes = [
            VersionChange(
                action=VersionAction.UPDATE,
                entity_type="document",
                entity_id=123
            )
        ]
        
        info = VersionInfo(
            version_id=1,
            version_number="1.2.0",
            description="Test version",
            created_at=datetime.now(timezone.utc),
            created_by="test_user",
            status=VersionStatus.ACTIVE,
            changes=changes,
            snapshot_size=1024,
            is_current=True
        )
        
        assert info.version_id == 1
        assert info.version_number == "1.2.0"
        assert info.description == "Test version"
        assert info.created_by == "test_user"
        assert info.status == VersionStatus.ACTIVE
        assert len(info.changes) == 1
        assert info.snapshot_size == 1024
        assert info.is_current is True


class TestKnowledgeBaseVersioning:
    """Test suite for KnowledgeBaseVersioning service"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_knowledge_base(self):
        """Mock knowledge base"""
        kb = Mock()
        kb.id = 1
        kb.name = "Test KB"
        kb.description = "Test knowledge base"
        kb.chunking_config = {}
        kb.embedding_config = {}
        return kb
    
    @pytest.fixture
    def mock_version(self):
        """Mock version"""
        version = Mock()
        version.id = 1
        version.knowledge_base_id = 1
        version.version = "1.0.0"
        version.description = "Initial version"
        version.created_by = "test_user"
        version.status = VersionStatus.ACTIVE.value
        version.is_current = True
        version.changes = "[]"
        version.metadata = {}
        version.created_at = datetime.now(timezone.utc)
        version.updated_at = None
        return version
    
    @pytest.fixture
    def versioning_service(self, mock_db):
        """Create versioning service with mocked dependencies"""
        with patch('app.services.versioning.VectorService') as mock_vector_service:
            with patch('app.services.versioning.DocumentIngestionService') as mock_doc_service:
                service = KnowledgeBaseVersioning(mock_db)
                return service
    
    def test_increment_version_number(self, versioning_service):
        """Test version number increment"""
        # Test normal increment
        assert versioning_service._increment_version_number("1.0.0") == "1.1.0"
        assert versioning_service._increment_version_number("1.5.0") == "1.6.0"
        assert versioning_service._increment_version_number("2.10.0") == "2.11.0"
        
        # Test invalid version format
        assert versioning_service._increment_version_number("invalid") == "1.0.0"
        assert versioning_service._increment_version_number("") == "1.0.0"
    
    def test_get_current_version(self, versioning_service, mock_db, mock_version):
        """Test getting current version"""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_version
        
        current = versioning_service._get_current_version(1)
        
        assert current == mock_version
        mock_db.query.assert_called_once()
    
    def test_get_current_version_none(self, versioning_service, mock_db):
        """Test getting current version when none exists"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        current = versioning_service._get_current_version(1)
        
        assert current is None
    
    @pytest.mark.asyncio
    async def test_create_version(self, versioning_service, mock_db, mock_knowledge_base):
        """Test creating a new version"""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = mock_knowledge_base
        
        # Mock current version
        mock_current_version = Mock()
        mock_current_version.version = "1.0.0"
        mock_current_version.status = VersionStatus.ACTIVE.value
        mock_current_version.is_current = True
        
        with patch.object(versioning_service, '_get_current_version', return_value=mock_current_version):
            with patch.object(versioning_service, '_create_snapshot', return_value=True):
                # Create new version
                changes = [
                    VersionChange(
                        action=VersionAction.UPDATE,
                        entity_type="document",
                        entity_id=123
                    )
                ]
                
                version = await versioning_service.create_version(
                    knowledge_base_id=1,
                    description="Test version",
                    created_by="test_user",
                    changes=changes
                )
                
                # Verify version was created
                mock_db.add.assert_called_once()
                mock_db.commit.assert_called()
                
                # Verify current version was updated
                assert mock_current_version.status == VersionStatus.INACTIVE.value
                assert mock_current_version.is_current is False
    
    @pytest.mark.asyncio
    async def test_create_version_no_knowledge_base(self, versioning_service, mock_db):
        """Test creating version for non-existent knowledge base"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ValueError, match="Knowledge base 1 not found"):
            await versioning_service.create_version(
                knowledge_base_id=1,
                description="Test version",
                created_by="test_user"
            )
    
    def test_get_version_history(self, versioning_service, mock_db):
        """Test getting version history"""
        # Setup mock versions
        mock_versions = []
        for i in range(3):
            version = Mock()
            version.id = i + 1
            version.version = f"1.{i}.0"
            version.description = f"Version {i}"
            version.created_by = "test_user"
            version.status = VersionStatus.ACTIVE.value if i == 2 else VersionStatus.INACTIVE.value
            version.is_current = i == 2
            version.changes = "[]"
            version.created_at = datetime.now(timezone.utc)
            mock_versions.append(version)
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_versions
        
        # Mock snapshots
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Get version history
        history = versioning_service.get_version_history(1)
        
        assert len(history) == 3
        assert history[0].version_number == "1.0.0"
        assert history[0].description == "Version 0"
        assert history[0].created_by == "test_user"
        assert history[2].is_current is True
    
    def test_get_version_history_empty(self, versioning_service, mock_db):
        """Test getting version history when empty"""
        mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        
        history = versioning_service.get_version_history(1)
        
        assert len(history) == 0
    
    def test_get_version_statistics(self, versioning_service, mock_db):
        """Test getting version statistics"""
        # Setup mock versions
        mock_versions = []
        for i in range(5):
            version = Mock()
            version.id = i + 1
            version.version = f"1.{i}.0"
            version.status = VersionStatus.ACTIVE.value if i == 4 else VersionStatus.INACTIVE.value
            version.is_current = i == 4
            version.description = f"Version {i}"
            version.created_at = datetime.now(timezone.utc)
            mock_versions.append(version)
        
        mock_db.query.return_value.filter.return_value.all.return_value = mock_versions
        
        # Mock snapshots
        mock_snapshot = Mock()
        mock_snapshot.snapshot_data = "test data"
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_snapshot]
        
        # Get statistics
        stats = versioning_service.get_version_statistics(1)
        
        assert stats["total_versions"] == 5
        assert stats["current_version"]["version"] == "1.4.0"
        assert stats["oldest_version"]["version"] == "1.0.0"
        assert stats["newest_version"]["version"] == "1.4.0"
        assert stats["status_distribution"]["active"] == 1
        assert stats["status_distribution"]["inactive"] == 4
        assert stats["total_snapshots"] == 1
        assert stats["total_snapshot_size"] == 9  # len("test data")
    
    def test_get_version_statistics_empty(self, versioning_service, mock_db):
        """Test getting statistics when no versions exist"""
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        stats = versioning_service.get_version_statistics(1)
        
        assert stats["total_versions"] == 0
        assert stats["current_version"] is None
        assert stats["oldest_version"] is None
        assert stats["newest_version"] is None
        assert stats["status_distribution"] == {}
        assert stats["total_snapshots"] == 0
        assert stats["total_snapshot_size"] == 0
    
    def test_compare_versions(self, versioning_service, mock_db):
        """Test comparing two versions"""
        # Setup mock versions
        version1 = Mock()
        version1.id = 1
        version1.version = "1.0.0"
        version1.description = "Version 1"
        version1.created_at = datetime.now(timezone.utc)
        version1.created_by = "test_user"
        
        version2 = Mock()
        version2.id = 2
        version2.version = "1.1.0"
        version2.description = "Version 2"
        version2.created_at = datetime.now(timezone.utc)
        version2.created_by = "test_user"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [version1, version2]
        
        # Setup mock snapshots
        snapshot1 = Mock()
        snapshot1.snapshot_data = json.dumps({"documents": []})
        
        snapshot2 = Mock()
        snapshot2.snapshot_data = json.dumps({"documents": []})
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [version1, version2, snapshot1, snapshot2]
        
        # Mock the differences computation
        with patch.object(versioning_service, '_compute_differences', return_value={"test": "diff"}):
            comparison = versioning_service.compare_versions(1, 1, 2)
            
            assert comparison["version1"]["version"] == "1.0.0"
            assert comparison["version2"]["version"] == "1.1.0"
            assert comparison["differences"]["test"] == "diff"
    
    def test_compare_versions_not_found(self, versioning_service, mock_db):
        """Test comparing versions when one doesn't exist"""
        mock_db.query.return_value.filter.return_value.first.side_effect = [None, None]
        
        with pytest.raises(ValueError, match="One or both versions not found"):
            versioning_service.compare_versions(1, 1, 2)
    
    def test_compute_differences(self, versioning_service):
        """Test computing differences between snapshots"""
        # Setup mock snapshots
        snapshot1 = Mock()
        snapshot1.snapshot_data = json.dumps({
            "documents": [
                {"id": 1, "title": "Doc 1", "content_type": "text/plain", "chunks": []},
                {"id": 2, "title": "Doc 2", "content_type": "text/plain", "chunks": []}
            ]
        })
        
        snapshot2 = Mock()
        snapshot2.snapshot_data = json.dumps({
            "documents": [
                {"id": 1, "title": "Doc 1 Updated", "content_type": "text/plain", "chunks": []},
                {"id": 3, "title": "Doc 3", "content_type": "text/plain", "chunks": []}
            ]
        })
        
        # Compute differences
        diff = versioning_service._compute_differences(snapshot1, snapshot2)
        
        assert diff["summary"]["documents_added"] == 1
        assert diff["summary"]["documents_removed"] == 1
        assert diff["summary"]["documents_modified"] == 1
        assert diff["summary"]["total_changes"] == 3
        
        # Check specific changes
        assert len(diff["documents"]["added"]) == 1
        assert diff["documents"]["added"][0]["id"] == 3
        
        assert len(diff["documents"]["removed"]) == 1
        assert diff["documents"]["removed"][0]["id"] == 2
        
        assert len(diff["documents"]["modified"]) == 1
        assert diff["documents"]["modified"][0]["id"] == 1
    
    def test_compute_differences_invalid_data(self, versioning_service):
        """Test computing differences with invalid snapshot data"""
        snapshot1 = Mock()
        snapshot1.snapshot_data = "invalid json"
        
        snapshot2 = Mock()
        snapshot2.snapshot_data = json.dumps({"documents": []})
        
        diff = versioning_service._compute_differences(snapshot1, snapshot2)
        
        assert "error" in diff
        assert diff["error"] == "Invalid snapshot data"
    
    def test_compute_differences_missing_snapshots(self, versioning_service):
        """Test computing differences with missing snapshots"""
        diff = versioning_service._compute_differences(None, None)
        
        assert "error" in diff
        assert diff["error"] == "One or both snapshots not found"
    
    @pytest.mark.asyncio
    async def test_archive_version(self, versioning_service, mock_db):
        """Test archiving a version"""
        # Setup mock version
        version = Mock()
        version.id = 1
        version.is_current = False
        version.status = VersionStatus.INACTIVE.value
        version.metadata = {}
        
        mock_db.query.return_value.filter.return_value.first.return_value = version
        
        # Archive version
        success = await versioning_service.archive_version(1, "Test reason")
        
        assert success is True
        assert version.status == VersionStatus.ARCHIVED.value
        assert version.metadata["archived_reason"] == "Test reason"
        assert "archived_at" in version.metadata
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_archive_version_current(self, versioning_service, mock_db):
        """Test archiving current version (should fail)"""
        # Setup mock current version
        version = Mock()
        version.id = 1
        version.is_current = True
        
        mock_db.query.return_value.filter.return_value.first.return_value = version
        
        # Archive version should fail
        with pytest.raises(ValueError, match="Cannot archive the current version"):
            await versioning_service.archive_version(1)
    
    @pytest.mark.asyncio
    async def test_archive_version_not_found(self, versioning_service, mock_db):
        """Test archiving non-existent version"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        success = await versioning_service.archive_version(1)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_delete_version_soft(self, versioning_service, mock_db):
        """Test soft delete of version"""
        # Setup mock version
        version = Mock()
        version.id = 1
        version.is_current = False
        version.status = VersionStatus.INACTIVE.value
        version.metadata = {}
        
        mock_db.query.return_value.filter.return_value.first.return_value = version
        
        # Delete version (soft)
        success = await versioning_service.delete_version(1, permanent=False)
        
        assert success is True
        assert version.status == VersionStatus.DELETED.value
        assert "deleted_at" in version.metadata
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_version_hard(self, versioning_service, mock_db):
        """Test hard delete of version"""
        # Setup mock version and snapshot
        version = Mock()
        version.id = 1
        version.is_current = False
        
        snapshot = Mock()
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [version, snapshot]
        
        # Delete version (hard)
        success = await versioning_service.delete_version(1, permanent=True)
        
        assert success is True
        # Should delete both snapshot and version
        assert mock_db.delete.call_count == 2
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_version_current(self, versioning_service, mock_db):
        """Test deleting current version (should fail)"""
        # Setup mock current version
        version = Mock()
        version.id = 1
        version.is_current = True
        
        mock_db.query.return_value.filter.return_value.first.return_value = version
        
        # Delete version should fail
        with pytest.raises(ValueError, match="Cannot delete the current version"):
            await versioning_service.delete_version(1)
    
    @pytest.mark.asyncio
    async def test_create_snapshot(self, versioning_service, mock_db):
        """Test creating a snapshot"""
        # Setup mock version
        version = Mock()
        version.id = 1
        version.knowledge_base_id = 1
        version.knowledge_base = Mock()
        version.knowledge_base.name = "Test KB"
        version.knowledge_base.description = "Test"
        version.knowledge_base.chunking_config = {}
        version.knowledge_base.embedding_config = {}
        
        mock_db.query.return_value.filter.return_value.first.return_value = version
        
        # Setup mock documents and chunks
        mock_doc = Mock()
        mock_doc.id = 1
        mock_doc.title = "Test Doc"
        mock_doc.content = "Test content"
        mock_doc.content_type = "text/plain"
        mock_doc.source_url = None
        mock_doc.metadata = {}
        mock_doc.processing_status = "completed"
        mock_doc.content_hash = "abc123"
        mock_doc.version = "1.0.0"
        
        mock_chunk = Mock()
        mock_chunk.id = 1
        mock_chunk.chunk_text = "Test chunk"
        mock_chunk.chunk_index = 0
        mock_chunk.metadata = {}
        mock_chunk.vector_id = "vector_1"
        
        mock_db.query.return_value.filter.return_value.all.side_effect = [[mock_doc], [mock_chunk]]
        
        # Create snapshot
        success = await versioning_service._create_snapshot(1)
        
        assert success is True
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_snapshot_existing(self, versioning_service, mock_db):
        """Test creating snapshot when one already exists"""
        # Setup mocks
        version = Mock()
        version.id = 1
        
        existing_snapshot = Mock()
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [version, existing_snapshot]
        
        # Create snapshot
        success = await versioning_service._create_snapshot(1)
        
        assert success is True
        # Should not add new snapshot
        mock_db.add.assert_not_called()


class TestVersioningUtilities:
    """Test suite for versioning utility functions"""
    
    @pytest.mark.asyncio
    async def test_auto_version_on_changes(self):
        """Test auto-versioning utility function"""
        mock_db = Mock(spec=Session)
        
        changes = [
            VersionChange(
                action=VersionAction.UPDATE,
                entity_type="document",
                entity_id=123
            )
        ]
        
        with patch('app.services.versioning.KnowledgeBaseVersioning') as mock_versioning_class:
            mock_versioning = Mock()
            mock_version = Mock()
            mock_version.id = 1
            mock_versioning.create_version.return_value = mock_version
            mock_versioning_class.return_value = mock_versioning
            
            version = await auto_version_on_changes(
                knowledge_base_id=1,
                changes=changes,
                created_by="test_user",
                description="Test auto-version",
                db=mock_db
            )
            
            assert version.id == 1
            mock_versioning.create_version.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auto_version_on_changes_auto_description(self):
        """Test auto-versioning with auto-generated description"""
        mock_db = Mock(spec=Session)
        
        changes = [
            VersionChange(action=VersionAction.UPDATE, entity_type="document", entity_id=123),
            VersionChange(action=VersionAction.CREATE, entity_type="document", entity_id=124)
        ]
        
        with patch('app.services.versioning.KnowledgeBaseVersioning') as mock_versioning_class:
            mock_versioning = Mock()
            mock_version = Mock()
            mock_versioning.create_version.return_value = mock_version
            mock_versioning_class.return_value = mock_versioning
            
            await auto_version_on_changes(
                knowledge_base_id=1,
                changes=changes,
                created_by="test_user",
                db=mock_db
            )
            
            # Should call create_version with auto-generated description
            call_args = mock_versioning.create_version.call_args
            assert "Auto-version:" in call_args[1]['description']
            assert "update" in call_args[1]['description']
            assert "create" in call_args[1]['description']
    
    def test_create_change_for_document_update(self):
        """Test utility function for creating document update change"""
        from app.services.versioning import create_change_for_document_update
        
        old_data = {"title": "Old Title", "content": "Old content"}
        new_data = {"title": "New Title", "content": "New content"}
        
        change = create_change_for_document_update(123, old_data, new_data)
        
        assert change.action == VersionAction.UPDATE
        assert change.entity_type == "document"
        assert change.entity_id == 123
        assert change.old_data == old_data
        assert change.new_data == new_data
    
    def test_create_change_for_document_creation(self):
        """Test utility function for creating document creation change"""
        from app.services.versioning import create_change_for_document_creation
        
        document_data = {"title": "New Document", "content": "New content"}
        
        change = create_change_for_document_creation(123, document_data)
        
        assert change.action == VersionAction.CREATE
        assert change.entity_type == "document"
        assert change.entity_id == 123
        assert change.old_data is None
        assert change.new_data == document_data
    
    def test_create_change_for_document_deletion(self):
        """Test utility function for creating document deletion change"""
        from app.services.versioning import create_change_for_document_deletion
        
        document_data = {"title": "Deleted Document", "content": "Deleted content"}
        
        change = create_change_for_document_deletion(123, document_data)
        
        assert change.action == VersionAction.DELETE
        assert change.entity_type == "document"
        assert change.entity_id == 123
        assert change.old_data == document_data
        assert change.new_data is None


class TestVersioningIntegration:
    """Integration tests for versioning system"""
    
    @pytest.mark.asyncio
    async def test_full_versioning_workflow(self):
        """Test complete versioning workflow"""
        mock_db = Mock(spec=Session)
        
        # This would be a full integration test
        # For now, we'll create a simplified mock implementation
        
        with patch('app.services.versioning.VectorService') as mock_vector_service:
            with patch('app.services.versioning.DocumentIngestionService') as mock_doc_service:
                # Setup mocks
                mock_kb = Mock()
                mock_kb.id = 1
                mock_kb.name = "Test KB"
                
                mock_db.query.return_value.filter.return_value.first.return_value = mock_kb
                
                versioning = KnowledgeBaseVersioning(mock_db)
                
                # Test version creation
                with patch.object(versioning, '_create_snapshot', return_value=True):
                    version = await versioning.create_version(
                        knowledge_base_id=1,
                        description="Test version",
                        created_by="test_user"
                    )
                    
                    # Verify version was created
                    mock_db.add.assert_called_once()
                    mock_db.commit.assert_called()


class TestVersioningErrorHandling:
    """Test error handling in versioning system"""
    
    @pytest.mark.asyncio
    async def test_rollback_to_nonexistent_version(self):
        """Test rollback to non-existent version"""
        mock_db = Mock(spec=Session)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        versioning = KnowledgeBaseVersioning(mock_db)
        
        with pytest.raises(ValueError, match="Version 999 not found"):
            await versioning.rollback_to_version(
                knowledge_base_id=1,
                target_version_id=999,
                created_by="test_user"
            )
    
    @pytest.mark.asyncio
    async def test_rollback_without_snapshot(self):
        """Test rollback when no snapshot exists"""
        mock_db = Mock(spec=Session)
        
        # Setup mock version without snapshot
        mock_version = Mock()
        mock_version.id = 1
        mock_version.version = "1.0.0"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [mock_version, None]
        
        versioning = KnowledgeBaseVersioning(mock_db)
        
        with pytest.raises(ValueError, match="No snapshot found for version 1"):
            await versioning.rollback_to_version(
                knowledge_base_id=1,
                target_version_id=1,
                created_by="test_user"
            )
    
    @pytest.mark.asyncio
    async def test_create_snapshot_invalid_version(self):
        """Test creating snapshot for invalid version"""
        mock_db = Mock(spec=Session)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        versioning = KnowledgeBaseVersioning(mock_db)
        
        success = await versioning._create_snapshot(999)
        
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__])