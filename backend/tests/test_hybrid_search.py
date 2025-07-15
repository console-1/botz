import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.orm import Session

from app.services.hybrid_search import (
    HybridSearchService, KeywordSearchEngine, HybridSearchConfig,
    SearchMode, SearchWeights, SearchResult, quick_search
)
from app.models.client import Client, KnowledgeBase, Document, DocumentChunk


class TestSearchWeights:
    """Test suite for SearchWeights"""
    
    def test_default_weights(self):
        """Test default weight initialization"""
        weights = SearchWeights()
        assert weights.semantic_weight == 0.7
        assert weights.keyword_weight == 0.3
        assert abs(weights.semantic_weight + weights.keyword_weight - 1.0) < 0.001
    
    def test_custom_weights_normalization(self):
        """Test weight normalization"""
        weights = SearchWeights(semantic_weight=0.8, keyword_weight=0.4)
        # Should normalize to sum to 1.0
        assert abs(weights.semantic_weight + weights.keyword_weight - 1.0) < 0.001
        assert weights.semantic_weight == 0.8 / 1.2
        assert weights.keyword_weight == 0.4 / 1.2
    
    def test_zero_weights(self):
        """Test handling of zero weights"""
        weights = SearchWeights(semantic_weight=0.0, keyword_weight=0.0)
        # Should handle zero division gracefully
        assert weights.semantic_weight == 0.0
        assert weights.keyword_weight == 0.0


class TestHybridSearchConfig:
    """Test suite for HybridSearchConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = HybridSearchConfig()
        assert config.mode == SearchMode.HYBRID
        assert config.weights.semantic_weight == 0.7
        assert config.weights.keyword_weight == 0.3
        assert config.semantic_threshold == 0.7
        assert config.keyword_threshold == 0.1
        assert config.max_results == 20
        assert config.enable_reranking is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        weights = SearchWeights(semantic_weight=0.6, keyword_weight=0.4)
        config = HybridSearchConfig(
            mode=SearchMode.SEMANTIC_ONLY,
            weights=weights,
            semantic_threshold=0.8,
            max_results=50
        )
        assert config.mode == SearchMode.SEMANTIC_ONLY
        assert config.weights.semantic_weight == 0.6
        assert config.weights.keyword_weight == 0.4
        assert config.semantic_threshold == 0.8
        assert config.max_results == 50


class TestSearchResult:
    """Test suite for SearchResult"""
    
    def test_search_result_creation(self):
        """Test SearchResult creation"""
        result = SearchResult(
            chunk_id=1,
            document_id=10,
            document_title="Test Document",
            chunk_index=0,
            text="Test content",
            combined_score=0.85,
            semantic_score=0.8,
            keyword_score=0.9,
            metadata={"section": "intro"},
            match_type="hybrid"
        )
        
        assert result.chunk_id == 1
        assert result.document_id == 10
        assert result.combined_score == 0.85
        assert result.match_type == "hybrid"
    
    def test_to_dict(self):
        """Test SearchResult to_dict conversion"""
        result = SearchResult(
            chunk_id=1,
            document_id=10,
            document_title="Test Document",
            chunk_index=0,
            text="Test content",
            combined_score=0.85,
            semantic_score=0.8,
            keyword_score=0.9,
            metadata={"section": "intro"},
            match_type="hybrid"
        )
        
        result_dict = result.to_dict()
        assert result_dict["chunk_id"] == 1
        assert result_dict["combined_score"] == 0.85
        assert result_dict["match_type"] == "hybrid"


class TestKeywordSearchEngine:
    """Test suite for KeywordSearchEngine"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def keyword_engine(self, mock_db):
        """Create KeywordSearchEngine instance"""
        return KeywordSearchEngine(mock_db)
    
    def test_clean_query(self, keyword_engine):
        """Test query cleaning"""
        # Test normal query
        assert keyword_engine._clean_query("How to install software?") == "how to install software"
        
        # Test query with special characters
        assert keyword_engine._clean_query("API & SDK documentation!") == "api sdk documentation"
        
        # Test query with multiple spaces
        assert keyword_engine._clean_query("  multiple   spaces  ") == "multiple spaces"
    
    def test_calculate_keyword_score(self, keyword_engine):
        """Test keyword score calculation"""
        result = {
            'text': "This is a test document about software installation",
            'document_title': "Software Installation Guide",
            'match_type': 'exact_phrase'
        }
        
        # Test exact phrase match
        score = keyword_engine._calculate_keyword_score(result, "software installation")
        assert score > 0.7  # Should be high for exact phrase match
        
        # Test partial match
        score = keyword_engine._calculate_keyword_score(result, "installation")
        assert 0.3 < score < 0.7  # Should be moderate
    
    def test_highlight_matches(self, keyword_engine):
        """Test text highlighting"""
        text = "This is a test document about software installation"
        query = "software installation"
        
        highlighted = keyword_engine.highlight_matches(text, query)
        assert "<mark>" in highlighted
        assert "</mark>" in highlighted
        assert "software" in highlighted
        assert "installation" in highlighted
    
    def test_search_empty_query(self, keyword_engine):
        """Test search with empty query"""
        results = keyword_engine.search("", "test-client")
        assert results == []
    
    def test_search_with_mock_data(self, keyword_engine, mock_db):
        """Test search with mocked database data"""
        # Mock database query results
        mock_chunk = Mock()
        mock_chunk.id = 1
        mock_chunk.document_id = 10
        mock_chunk.chunk_index = 0
        mock_chunk.chunk_text = "How to install software on your system"
        mock_chunk.metadata = {"section": "installation"}
        mock_chunk.vector_id = "vector-1"
        mock_chunk.title = "Installation Guide"
        mock_chunk.created_at = "2024-01-01"
        
        mock_db.query.return_value.join.return_value.filter.return_value.filter.return_value.limit.return_value.all.return_value = [mock_chunk]
        
        results = keyword_engine.search("install software", "test-client")
        
        # Should have some results (exact implementation depends on mocking)
        assert isinstance(results, list)


class TestHybridSearchService:
    """Test suite for HybridSearchService"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        db = Mock(spec=Session)
        return db
    
    @pytest.fixture
    def mock_vector_service(self):
        """Mock vector service"""
        vector_service = Mock()
        vector_service.search_similar_chunks.return_value = [
            {
                'chunk_id': 1,
                'document_id': 10,
                'chunk_index': 0,
                'text': 'How to install software',
                'metadata': {'section': 'installation'},
                'vector_id': 'vector-1',
                'document_title': 'Installation Guide',
                'score': 0.85
            }
        ]
        return vector_service
    
    @pytest.fixture
    def mock_keyword_engine(self):
        """Mock keyword search engine"""
        keyword_engine = Mock()
        keyword_engine.search.return_value = [
            {
                'chunk_id': 1,
                'document_id': 10,
                'chunk_index': 0,
                'text': 'How to install software',
                'metadata': {'section': 'installation'},
                'vector_id': 'vector-1',
                'document_title': 'Installation Guide',
                'score': 0.75
            }
        ]
        keyword_engine.highlight_matches.return_value = 'How to <mark>install</mark> software'
        return keyword_engine
    
    @pytest.fixture
    def search_service(self, mock_db, mock_vector_service, mock_keyword_engine):
        """Create HybridSearchService with mocked dependencies"""
        service = HybridSearchService(mock_db)
        service.vector_service = mock_vector_service
        service.keyword_engine = mock_keyword_engine
        return service
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, search_service):
        """Test search with empty query"""
        results = await search_service.search("", "test-client")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_short_query(self, search_service):
        """Test search with very short query"""
        config = HybridSearchConfig(min_query_length=3)
        results = await search_service.search("hi", "test-client", config)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_determine_search_mode(self, search_service):
        """Test search mode determination"""
        config = HybridSearchConfig(mode=SearchMode.AUTO)
        
        # Test quoted query
        mode = search_service._determine_search_mode('"exact phrase"', config)
        assert mode == SearchMode.KEYWORD_ONLY
        
        # Test short query
        mode = search_service._determine_search_mode('help', config)
        assert mode == SearchMode.SEMANTIC_ONLY
        
        # Test technical query
        mode = search_service._determine_search_mode('API 404 error', config)
        assert mode == SearchMode.HYBRID
        
        # Test normal query
        mode = search_service._determine_search_mode('how to install software', config)
        assert mode == SearchMode.HYBRID
    
    @pytest.mark.asyncio
    async def test_semantic_only_search(self, search_service, mock_vector_service):
        """Test semantic-only search"""
        config = HybridSearchConfig(mode=SearchMode.SEMANTIC_ONLY)
        
        results = await search_service.search("install software", "test-client", config)
        
        # Should call vector service but not keyword engine
        mock_vector_service.search_similar_chunks.assert_called_once()
        assert len(results) > 0
        assert all(r.match_type in ['semantic', 'hybrid'] for r in results)
    
    @pytest.mark.asyncio
    async def test_keyword_only_search(self, search_service, mock_keyword_engine):
        """Test keyword-only search"""
        config = HybridSearchConfig(mode=SearchMode.KEYWORD_ONLY)
        
        results = await search_service.search("install software", "test-client", config)
        
        # Should call keyword engine but not vector service
        mock_keyword_engine.search.assert_called_once()
        assert len(results) > 0
        assert all(r.match_type in ['keyword', 'hybrid'] for r in results)
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, search_service, mock_vector_service, mock_keyword_engine):
        """Test hybrid search combining both approaches"""
        config = HybridSearchConfig(mode=SearchMode.HYBRID)
        
        results = await search_service.search("install software", "test-client", config)
        
        # Should call both services
        mock_vector_service.search_similar_chunks.assert_called_once()
        mock_keyword_engine.search.assert_called_once()
        
        assert len(results) > 0
        # Should have combined scores
        for result in results:
            assert result.combined_score > 0
            assert result.match_type in ['semantic', 'keyword', 'hybrid']
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_service, mock_vector_service, mock_keyword_engine):
        """Test search with knowledge base and document filters"""
        config = HybridSearchConfig()
        
        results = await search_service.search(
            "install software", 
            "test-client", 
            config,
            knowledge_base_id=1,
            document_id=10
        )
        
        # Verify filters were passed to both services
        mock_vector_service.search_similar_chunks.assert_called_once()
        call_args = mock_vector_service.search_similar_chunks.call_args
        assert call_args[1]['knowledge_base_id'] == 1
        assert call_args[1]['document_id'] == 10
        
        mock_keyword_engine.search.assert_called_once()
        call_args = mock_keyword_engine.search.call_args
        assert call_args[1]['knowledge_base_id'] == 1
        assert call_args[1]['document_id'] == 10
    
    @pytest.mark.asyncio
    async def test_search_with_analytics(self, search_service):
        """Test search with analytics"""
        config = HybridSearchConfig()
        
        results, analytics = await search_service.search_with_analytics(
            "install software", 
            "test-client", 
            config
        )
        
        # Verify analytics structure
        assert 'query' in analytics
        assert 'total_results' in analytics
        assert 'search_time_ms' in analytics
        assert 'semantic_results' in analytics
        assert 'keyword_results' in analytics
        assert 'hybrid_results' in analytics
        assert 'score_distribution' in analytics
        
        assert analytics['query'] == "install software"
        assert analytics['total_results'] == len(results)
        assert analytics['search_time_ms'] > 0
    
    def test_apply_boosts(self, search_service):
        """Test score boosting functionality"""
        config = HybridSearchConfig()
        
        result = SearchResult(
            chunk_id=1,
            document_id=10,
            document_title="Installation Guide",
            chunk_index=0,
            text="How to install software on your system",
            combined_score=0.7,
            semantic_score=0.75,
            keyword_score=0.65,
            metadata={},
            match_type="hybrid"
        )
        
        # Test exact match boost
        boosted_score = search_service._apply_boosts(result, "install software", config)
        assert boosted_score > result.combined_score
        
        # Test title match boost
        boosted_score = search_service._apply_boosts(result, "installation", config)
        assert boosted_score > result.combined_score
    
    def test_combine_results(self, search_service):
        """Test combining semantic and keyword results"""
        config = HybridSearchConfig()
        
        semantic_results = [
            {
                'chunk_id': 1,
                'document_id': 10,
                'document_title': 'Guide 1',
                'chunk_index': 0,
                'text': 'Content 1',
                'metadata': {},
                'vector_id': 'v1',
                'score': 0.8
            }
        ]
        
        keyword_results = [
            {
                'chunk_id': 1,  # Same chunk
                'document_id': 10,
                'document_title': 'Guide 1',
                'chunk_index': 0,
                'text': 'Content 1',
                'metadata': {},
                'vector_id': 'v1',
                'score': 0.7
            },
            {
                'chunk_id': 2,  # Different chunk
                'document_id': 11,
                'document_title': 'Guide 2',
                'chunk_index': 0,
                'text': 'Content 2',
                'metadata': {},
                'vector_id': 'v2',
                'score': 0.6
            }
        ]
        
        combined = search_service._combine_results(semantic_results, keyword_results, config, "test")
        
        assert len(combined) == 2  # Should have 2 unique chunks
        
        # First result should be hybrid (appeared in both)
        hybrid_result = next(r for r in combined if r.chunk_id == 1)
        assert hybrid_result.match_type == 'hybrid'
        assert hybrid_result.semantic_score == 0.8
        assert hybrid_result.keyword_score == 0.7
        
        # Second result should be keyword-only
        keyword_result = next(r for r in combined if r.chunk_id == 2)
        assert keyword_result.match_type == 'keyword'
        assert keyword_result.semantic_score == 0.0
        assert keyword_result.keyword_score == 0.6
    
    def test_analyze_score_distribution(self, search_service):
        """Test score distribution analysis"""
        results = [
            SearchResult(1, 10, "Doc 1", 0, "Text 1", 0.95, 0.9, 0.8, {}, match_type="hybrid"),
            SearchResult(2, 11, "Doc 2", 0, "Text 2", 0.75, 0.7, 0.8, {}, match_type="hybrid"),
            SearchResult(3, 12, "Doc 3", 0, "Text 3", 0.55, 0.5, 0.6, {}, match_type="hybrid"),
            SearchResult(4, 13, "Doc 4", 0, "Text 4", 0.35, 0.3, 0.4, {}, match_type="hybrid"),
        ]
        
        distribution = search_service._analyze_score_distribution(results)
        
        assert distribution['excellent'] == 1  # Score > 0.9
        assert distribution['good'] == 1       # Score 0.7-0.9
        assert distribution['fair'] == 1       # Score 0.5-0.7
        assert distribution['poor'] == 1       # Score < 0.5


class TestQuickSearch:
    """Test suite for quick search utility"""
    
    @pytest.mark.asyncio
    async def test_quick_search(self):
        """Test quick search utility function"""
        mock_db = Mock(spec=Session)
        
        with patch('app.services.hybrid_search.HybridSearchService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            mock_result = Mock()
            mock_result.to_dict.return_value = {'chunk_id': 1, 'text': 'test'}
            mock_service.search.return_value = [mock_result]
            
            results = await quick_search("test query", "test-client", mock_db, limit=5)
            
            assert len(results) == 1
            assert results[0]['chunk_id'] == 1
            mock_service.search.assert_called_once()


class TestHybridSearchIntegration:
    """Integration tests for hybrid search"""
    
    @pytest.mark.asyncio
    async def test_full_search_workflow(self):
        """Test complete search workflow"""
        # This would be a full integration test
        mock_db = Mock(spec=Session)
        
        with patch('app.services.hybrid_search.VectorService') as mock_vector_service:
            with patch('app.services.hybrid_search.KeywordSearchEngine') as mock_keyword_engine:
                # Setup mocks
                mock_vector_instance = Mock()
                mock_vector_instance.search_similar_chunks.return_value = [
                    {
                        'chunk_id': 1,
                        'document_id': 10,
                        'chunk_index': 0,
                        'text': 'How to install software',
                        'metadata': {'section': 'installation'},
                        'vector_id': 'vector-1',
                        'document_title': 'Installation Guide',
                        'score': 0.85
                    }
                ]
                mock_vector_service.return_value = mock_vector_instance
                
                mock_keyword_instance = Mock()
                mock_keyword_instance.search.return_value = [
                    {
                        'chunk_id': 1,
                        'document_id': 10,
                        'chunk_index': 0,
                        'text': 'How to install software',
                        'metadata': {'section': 'installation'},
                        'vector_id': 'vector-1',
                        'document_title': 'Installation Guide',
                        'score': 0.75
                    }
                ]
                mock_keyword_engine.return_value = mock_keyword_instance
                
                # Test the service
                service = HybridSearchService(mock_db)
                config = HybridSearchConfig(mode=SearchMode.HYBRID)
                
                results = await service.search("install software", "test-client", config)
                
                # Verify results
                assert len(results) > 0
                assert results[0].match_type == 'hybrid'
                assert results[0].combined_score > 0
                
                # Verify both services were called
                mock_vector_instance.search_similar_chunks.assert_called_once()
                mock_keyword_instance.search.assert_called_once()


class TestHybridSearchErrorHandling:
    """Test error handling in hybrid search"""
    
    @pytest.mark.asyncio
    async def test_semantic_search_failure(self):
        """Test handling of semantic search failures"""
        mock_db = Mock(spec=Session)
        
        with patch('app.services.hybrid_search.VectorService') as mock_vector_service:
            with patch('app.services.hybrid_search.KeywordSearchEngine') as mock_keyword_engine:
                # Setup semantic search to fail
                mock_vector_instance = Mock()
                mock_vector_instance.search_similar_chunks.side_effect = Exception("Vector search failed")
                mock_vector_service.return_value = mock_vector_instance
                
                # Setup keyword search to work
                mock_keyword_instance = Mock()
                mock_keyword_instance.search.return_value = [
                    {
                        'chunk_id': 1,
                        'document_id': 10,
                        'chunk_index': 0,
                        'text': 'How to install software',
                        'metadata': {'section': 'installation'},
                        'vector_id': 'vector-1',
                        'document_title': 'Installation Guide',
                        'score': 0.75
                    }
                ]
                mock_keyword_engine.return_value = mock_keyword_instance
                
                service = HybridSearchService(mock_db)
                config = HybridSearchConfig(mode=SearchMode.HYBRID)
                
                # Should not raise exception, should fallback to keyword only
                results = await service.search("install software", "test-client", config)
                
                assert len(results) > 0
                assert all(r.match_type == 'keyword' for r in results)
    
    @pytest.mark.asyncio
    async def test_keyword_search_failure(self):
        """Test handling of keyword search failures"""
        mock_db = Mock(spec=Session)
        
        with patch('app.services.hybrid_search.VectorService') as mock_vector_service:
            with patch('app.services.hybrid_search.KeywordSearchEngine') as mock_keyword_engine:
                # Setup semantic search to work
                mock_vector_instance = Mock()
                mock_vector_instance.search_similar_chunks.return_value = [
                    {
                        'chunk_id': 1,
                        'document_id': 10,
                        'chunk_index': 0,
                        'text': 'How to install software',
                        'metadata': {'section': 'installation'},
                        'vector_id': 'vector-1',
                        'document_title': 'Installation Guide',
                        'score': 0.85
                    }
                ]
                mock_vector_service.return_value = mock_vector_instance
                
                # Setup keyword search to fail
                mock_keyword_instance = Mock()
                mock_keyword_instance.search.side_effect = Exception("Keyword search failed")
                mock_keyword_engine.return_value = mock_keyword_instance
                
                service = HybridSearchService(mock_db)
                config = HybridSearchConfig(mode=SearchMode.HYBRID)
                
                # Should not raise exception, should fallback to semantic only
                results = await service.search("install software", "test-client", config)
                
                assert len(results) > 0
                assert all(r.match_type == 'semantic' for r in results)


if __name__ == "__main__":
    pytest.main([__file__])