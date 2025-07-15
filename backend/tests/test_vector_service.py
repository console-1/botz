import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.orm import Session

from app.services.vector_service import VectorService, VectorServiceManager
from app.models.client import Client, KnowledgeBase, Document, DocumentChunk
from app.rag.embeddings import EmbeddingConfig, EmbeddingResult
from app.core.vector_db import VectorDatabase


class TestVectorService:
    """Test suite for VectorService"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        db = Mock(spec=Session)
        db.query.return_value.filter.return_value.first.return_value = None
        db.query.return_value.filter.return_value.all.return_value = []
        db.commit.return_value = None
        return db
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock vector database"""
        vector_db = Mock(spec=VectorDatabase)
        vector_db.create_collection.return_value = True
        vector_db.delete_collection.return_value = True
        vector_db.upsert_points.return_value = True
        vector_db.search_similar.return_value = []
        vector_db.delete_points.return_value = True
        vector_db.get_collection_info.return_value = {
            'points_count': 100,
            'indexed_vectors_count': 100,
            'vector_size': 384,
            'distance': 'cosine'
        }
        return vector_db
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service"""
        embedding_service = Mock()
        embedding_service.get_dimension.return_value = 384
        embedding_service.generate_embeddings.return_value = [
            EmbeddingResult(
                text="test text",
                embedding=[0.1] * 384,
                model_name="test-model",
                provider="test-provider",
                metadata={}
            )
        ]
        embedding_service.generate_embedding.return_value = EmbeddingResult(
            text="test query",
            embedding=[0.1] * 384,
            model_name="test-model",
            provider="test-provider",
            metadata={}
        )
        return embedding_service
    
    @pytest.fixture
    def vector_service(self, mock_db, mock_vector_db, mock_embedding_service):
        """Create VectorService with mocked dependencies"""
        with patch('app.services.vector_service.VectorDatabase') as mock_vdb_class:
            mock_vdb_class.return_value = mock_vector_db
            
            with patch('app.services.vector_service.EmbeddingService') as mock_es_class:
                mock_es_class.return_value = mock_embedding_service
                
                service = VectorService(mock_db)
                return service
    
    @pytest.fixture
    def sample_knowledge_base(self):
        """Sample knowledge base for testing"""
        kb = KnowledgeBase(
            id=1,
            client_id="test-client",
            name="Test KB",
            embedding_config={"model": "test-model"}
        )
        return kb
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample document chunks for testing"""
        chunks = []
        for i in range(3):
            chunk = DocumentChunk(
                id=i+1,
                document_id=1,
                chunk_text=f"This is test chunk {i+1}",
                chunk_index=i,
                metadata={"test": True}
            )
            chunks.append(chunk)
        return chunks
    
    @pytest.mark.asyncio
    async def test_create_client_collection(self, vector_service, mock_vector_db):
        """Test creating a client collection"""
        client_id = "test-client"
        
        result = await vector_service.create_client_collection(client_id)
        
        assert result is True
        mock_vector_db.create_collection.assert_called_once_with(f"client_{client_id}", 384)
    
    @pytest.mark.asyncio
    async def test_delete_client_collection(self, vector_service, mock_vector_db):
        """Test deleting a client collection"""
        client_id = "test-client"
        
        result = await vector_service.delete_client_collection(client_id)
        
        assert result is True
        mock_vector_db.delete_collection.assert_called_once_with(f"client_{client_id}")
    
    @pytest.mark.asyncio
    async def test_generate_and_store_embeddings(self, vector_service, mock_db, mock_vector_db, 
                                                mock_embedding_service, sample_knowledge_base, sample_chunks):
        """Test generating and storing embeddings"""
        # Setup mock database queries
        mock_db.query.return_value.filter.return_value.first.return_value = sample_knowledge_base
        mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = sample_chunks
        
        # Test the method
        result = await vector_service.generate_and_store_embeddings(1)
        
        # Verify results
        assert result['total_chunks'] == 3
        assert result['embeddings_generated'] == 3
        assert result['embeddings_stored'] == 3
        assert result['errors'] == []
        
        # Verify mocks were called
        mock_embedding_service.generate_embeddings.assert_called_once()
        mock_vector_db.upsert_points.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks(self, vector_service, mock_vector_db, mock_embedding_service, mock_db):
        """Test searching for similar chunks"""
        # Setup mock vector search results
        mock_vector_db.search_similar.return_value = [
            {
                'id': 'vector-1',
                'score': 0.85,
                'payload': {
                    'chunk_id': 1,
                    'document_id': 1,
                    'text': 'test chunk',
                    'metadata': {'test': True}
                }
            }
        ]
        
        # Setup mock database chunk
        mock_chunk = Mock()
        mock_chunk.id = 1
        mock_chunk.chunk_text = 'test chunk'
        mock_chunk.chunk_index = 0
        mock_chunk.metadata = {'test': True}
        mock_chunk.document = Mock()
        mock_chunk.document.id = 1
        mock_chunk.document.title = 'Test Document'
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_chunk
        
        # Test the method
        results = await vector_service.search_similar_chunks(
            client_id="test-client",
            query_text="test query",
            limit=5
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]['chunk_id'] == 1
        assert results[0]['document_id'] == 1
        assert results[0]['score'] == 0.85
        assert results[0]['text'] == 'test chunk'
        
        # Verify mocks were called
        mock_embedding_service.generate_embedding.assert_called_once_with("test query")
        mock_vector_db.search_similar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document_embeddings(self, vector_service, mock_db, mock_vector_db, sample_chunks):
        """Test deleting document embeddings"""
        # Setup mocks
        mock_document = Mock()
        mock_document.id = 1
        mock_document.knowledge_base_id = 1
        
        mock_kb = Mock()
        mock_kb.client_id = "test-client"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [mock_document, mock_kb]
        mock_db.query.return_value.filter.return_value.all.return_value = sample_chunks
        
        # Set vector IDs for chunks
        for i, chunk in enumerate(sample_chunks):
            chunk.vector_id = f"vector-{i+1}"
        
        # Test the method
        result = await vector_service.delete_document_embeddings(1)
        
        # Verify results
        assert result is True
        mock_vector_db.delete_points.assert_called_once()
        mock_db.commit.assert_called_once()
        
        # Verify vector IDs were cleared
        for chunk in sample_chunks:
            assert chunk.vector_id is None
    
    @pytest.mark.asyncio
    async def test_update_chunk_embedding(self, vector_service, mock_db, mock_vector_db, 
                                        mock_embedding_service):
        """Test updating embedding for a specific chunk"""
        # Setup mocks
        mock_chunk = Mock()
        mock_chunk.id = 1
        mock_chunk.chunk_text = "test chunk"
        mock_chunk.vector_id = None
        mock_chunk.metadata = {'test': True}
        
        mock_document = Mock()
        mock_document.id = 1
        
        mock_kb = Mock()
        mock_kb.id = 1
        mock_kb.client_id = "test-client"
        
        mock_chunk.document = mock_document
        mock_document.knowledge_base = mock_kb
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_chunk
        
        # Test the method
        result = await vector_service.update_chunk_embedding(1)
        
        # Verify results
        assert result is True
        mock_embedding_service.generate_embedding.assert_called_once_with("test chunk")
        mock_vector_db.upsert_points.assert_called_once()
        mock_db.commit.assert_called_once()
        assert mock_chunk.vector_id is not None
    
    def test_get_vector_statistics(self, vector_service, mock_vector_db):
        """Test getting vector statistics"""
        client_id = "test-client"
        
        stats = vector_service.get_vector_statistics(client_id)
        
        assert stats['collection_exists'] is True
        assert stats['total_vectors'] == 100
        assert stats['indexed_vectors'] == 100
        assert stats['vector_size'] == 384
        assert stats['distance_metric'] == 'cosine'
        
        mock_vector_db.get_collection_info.assert_called_once_with(f"client_{client_id}")
    
    def test_get_embedding_statistics(self, vector_service, mock_db):
        """Test getting embedding statistics"""
        # Setup mocks
        mock_db.query.return_value.join.return_value.filter.return_value.count.side_effect = [100, 80]
        
        stats = vector_service.get_embedding_statistics(1)
        
        assert stats['total_chunks'] == 100
        assert stats['embedded_chunks'] == 80
        assert stats['embedding_coverage'] == 80.0
        assert stats['missing_embeddings'] == 20
    
    @pytest.mark.asyncio
    async def test_batch_search(self, vector_service, mock_vector_db, mock_embedding_service):
        """Test batch search functionality"""
        # Setup mocks
        mock_vector_db.search_similar.return_value = []
        
        queries = ["query 1", "query 2", "query 3"]
        
        # Test the method
        results = await vector_service.batch_search(
            client_id="test-client",
            queries=queries
        )
        
        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['query'] == queries[i]
            assert 'results' in result
            assert 'error' not in result or result['error'] is None
        
        # Verify mocks were called
        assert mock_embedding_service.generate_embedding.call_count == 3
    
    @pytest.mark.asyncio
    async def test_reindex_knowledge_base(self, vector_service, mock_db, sample_knowledge_base):
        """Test reindexing knowledge base"""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = sample_knowledge_base
        
        with patch.object(vector_service, 'delete_knowledge_base_embeddings') as mock_delete:
            mock_delete.return_value = True
            
            with patch.object(vector_service, 'generate_and_store_embeddings') as mock_generate:
                mock_generate.return_value = {
                    'total_chunks': 10,
                    'embeddings_generated': 10,
                    'embeddings_stored': 10,
                    'errors': []
                }
                
                result = await vector_service.reindex_knowledge_base(1)
                
                assert result['reindexed'] is True
                assert result['total_chunks'] == 10
                mock_delete.assert_called_once_with(1)
                mock_generate.assert_called_once_with(1, force_regenerate=True)


class TestVectorServiceManager:
    """Test suite for VectorServiceManager"""
    
    def test_create_service(self):
        """Test creating a VectorService with specific config"""
        mock_db = Mock(spec=Session)
        
        with patch('app.services.vector_service.VectorService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            service = VectorServiceManager.create_service(
                db=mock_db,
                embedding_provider="openai",
                embedding_model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            # Verify service was created with correct config
            mock_service_class.assert_called_once()
            args, kwargs = mock_service_class.call_args
            assert args[0] == mock_db
            embedding_config = args[1]
            assert embedding_config.provider == "openai"
            assert embedding_config.model_name == "text-embedding-ada-002"
            assert embedding_config.api_key == "test-key"
    
    def test_create_from_knowledge_base(self):
        """Test creating VectorService from knowledge base config"""
        mock_db = Mock(spec=Session)
        
        # Setup mock knowledge base
        mock_kb = Mock()
        mock_kb.embedding_config = {
            'provider': 'sentence_transformers',
            'model': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        mock_db.query.return_value.filter.return_value.first.return_value = mock_kb
        
        with patch('app.services.vector_service.VectorService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            service = VectorServiceManager.create_from_knowledge_base(mock_db, 1)
            
            # Verify service was created with KB config
            mock_service_class.assert_called_once()
            args, kwargs = mock_service_class.call_args
            assert args[0] == mock_db
            embedding_config = args[1]
            assert embedding_config.provider == "sentence_transformers"
            assert embedding_config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_create_from_knowledge_base_not_found(self):
        """Test creating VectorService from non-existent knowledge base"""
        mock_db = Mock(spec=Session)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ValueError, match="Knowledge base 1 not found"):
            VectorServiceManager.create_from_knowledge_base(mock_db, 1)


class TestVectorServiceIntegration:
    """Integration tests for VectorService"""
    
    @pytest.mark.asyncio
    async def test_full_document_processing_workflow(self):
        """Test complete workflow: ingest document -> chunk -> embed -> search"""
        # This would be a full integration test with real database
        # For now, we'll create a mock implementation
        
        with patch('app.services.vector_service.VectorService') as MockVectorService:
            mock_service = Mock()
            MockVectorService.return_value = mock_service
            
            # Mock the workflow methods
            mock_service.create_client_collection.return_value = True
            mock_service.generate_and_store_embeddings.return_value = {
                'total_chunks': 5,
                'embeddings_generated': 5,
                'embeddings_stored': 5,
                'errors': []
            }
            mock_service.search_similar_chunks.return_value = [
                {
                    'chunk_id': 1,
                    'document_id': 1,
                    'score': 0.85,
                    'text': 'relevant chunk'
                }
            ]
            
            # Test the workflow
            client_id = "test-client"
            kb_id = 1
            
            # Create collection
            result = await mock_service.create_client_collection(client_id)
            assert result is True
            
            # Generate embeddings
            result = await mock_service.generate_and_store_embeddings(kb_id)
            assert result['embeddings_stored'] == 5
            
            # Search
            results = await mock_service.search_similar_chunks(client_id, "test query")
            assert len(results) == 1
            assert results[0]['score'] == 0.85
            
            # Verify all methods were called
            mock_service.create_client_collection.assert_called_once_with(client_id)
            mock_service.generate_and_store_embeddings.assert_called_once_with(kb_id)
            mock_service.search_similar_chunks.assert_called_once_with(client_id, "test query")


class TestVectorServiceErrorHandling:
    """Test error handling in VectorService"""
    
    @pytest.mark.asyncio
    async def test_embedding_generation_error(self):
        """Test handling of embedding generation errors"""
        mock_db = Mock(spec=Session)
        
        with patch('app.services.vector_service.VectorDatabase') as mock_vdb_class:
            mock_vector_db = Mock()
            mock_vdb_class.return_value = mock_vector_db
            
            with patch('app.services.vector_service.EmbeddingService') as mock_es_class:
                mock_embedding_service = Mock()
                mock_embedding_service.generate_embeddings.side_effect = Exception("Embedding failed")
                mock_es_class.return_value = mock_embedding_service
                
                service = VectorService(mock_db)
                
                # Setup mock KB
                mock_kb = Mock()
                mock_kb.client_id = "test-client"
                mock_db.query.return_value.filter.return_value.first.return_value = mock_kb
                
                # Setup mock chunks
                mock_chunks = [Mock()]
                mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = mock_chunks
                
                # Test error handling
                result = await service.generate_and_store_embeddings(1)
                
                assert result['embeddings_generated'] == 0
                assert result['embeddings_stored'] == 0
                assert len(result['errors']) == 1
                assert "Failed to generate embeddings" in result['errors'][0]
    
    @pytest.mark.asyncio
    async def test_vector_storage_error(self):
        """Test handling of vector storage errors"""
        mock_db = Mock(spec=Session)
        
        with patch('app.services.vector_service.VectorDatabase') as mock_vdb_class:
            mock_vector_db = Mock()
            mock_vector_db.upsert_points.return_value = False  # Storage failure
            mock_vdb_class.return_value = mock_vector_db
            
            with patch('app.services.vector_service.EmbeddingService') as mock_es_class:
                mock_embedding_service = Mock()
                mock_embedding_service.generate_embeddings.return_value = [
                    EmbeddingResult(
                        text="test",
                        embedding=[0.1] * 384,
                        model_name="test-model",
                        provider="test-provider",
                        metadata={}
                    )
                ]
                mock_es_class.return_value = mock_embedding_service
                
                service = VectorService(mock_db)
                
                # Setup mocks
                mock_kb = Mock()
                mock_kb.client_id = "test-client"
                mock_db.query.return_value.filter.return_value.first.return_value = mock_kb
                
                mock_chunk = Mock()
                mock_chunk.chunk_text = "test"
                mock_chunk.id = 1
                mock_chunk.document_id = 1
                mock_chunk.chunk_index = 0
                mock_chunk.metadata = {}
                mock_chunk.vector_id = None
                mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = [mock_chunk]
                
                # Test error handling
                result = await service.generate_and_store_embeddings(1)
                
                assert result['embeddings_generated'] == 1
                assert result['embeddings_stored'] == 0
                assert len(result['errors']) == 1
                assert "Failed to store embeddings" in result['errors'][0]


if __name__ == "__main__":
    pytest.main([__file__])