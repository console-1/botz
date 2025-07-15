from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class VectorSearchRequest(BaseModel):
    """Request schema for vector search"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(5, ge=1, le=100, description="Maximum number of results")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    knowledge_base_id: Optional[int] = Field(None, description="Filter by knowledge base ID")
    document_id: Optional[int] = Field(None, description="Filter by document ID")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "limit": 5,
                "score_threshold": 0.7,
                "knowledge_base_id": 1
            }
        }


class VectorSearchResult(BaseModel):
    """Individual search result"""
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any]
    vector_id: str
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "chunk_id": 123,
                "document_id": 45,
                "document_title": "Installation Guide",
                "chunk_index": 2,
                "text": "To install the software, follow these steps...",
                "score": 0.85,
                "metadata": {"section": "installation"},
                "vector_id": "uuid-123",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_provider": "sentence_transformers"
            }
        }


class VectorSearchResponse(BaseModel):
    """Response schema for vector search"""
    query: str
    results: List[VectorSearchResult]
    total_results: int
    score_threshold: float
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "results": [
                    {
                        "chunk_id": 123,
                        "document_id": 45,
                        "document_title": "Installation Guide",
                        "chunk_index": 2,
                        "text": "To install the software, follow these steps...",
                        "score": 0.85,
                        "metadata": {"section": "installation"},
                        "vector_id": "uuid-123",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "embedding_provider": "sentence_transformers"
                    }
                ],
                "total_results": 1,
                "score_threshold": 0.7
            }
        }


class BatchSearchRequest(BaseModel):
    """Request schema for batch search"""
    queries: List[str] = Field(..., description="List of search queries")
    limit: int = Field(5, ge=1, le=50, description="Maximum number of results per query")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    knowledge_base_id: Optional[int] = Field(None, description="Filter by knowledge base ID")
    
    @validator('queries')
    def validate_queries(cls, v):
        if len(v) == 0:
            raise ValueError('At least one query is required')
        if len(v) > 50:
            raise ValueError('Maximum 50 queries per batch')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "queries": [
                    "How to install the software?",
                    "How to configure settings?",
                    "Troubleshooting steps"
                ],
                "limit": 3,
                "score_threshold": 0.7,
                "knowledge_base_id": 1
            }
        }


class BatchSearchResult(BaseModel):
    """Individual batch search result"""
    query: str
    results: List[VectorSearchResult]
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "results": [
                    {
                        "chunk_id": 123,
                        "document_id": 45,
                        "document_title": "Installation Guide",
                        "chunk_index": 2,
                        "text": "To install the software, follow these steps...",
                        "score": 0.85,
                        "metadata": {"section": "installation"},
                        "vector_id": "uuid-123",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "embedding_provider": "sentence_transformers"
                    }
                ],
                "error": None
            }
        }


class BatchSearchResponse(BaseModel):
    """Response schema for batch search"""
    results: List[BatchSearchResult]
    total_queries: int
    score_threshold: float
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "query": "How to install the software?",
                        "results": [
                            {
                                "chunk_id": 123,
                                "document_id": 45,
                                "document_title": "Installation Guide",
                                "chunk_index": 2,
                                "text": "To install the software, follow these steps...",
                                "score": 0.85,
                                "metadata": {"section": "installation"},
                                "vector_id": "uuid-123",
                                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                                "embedding_provider": "sentence_transformers"
                            }
                        ],
                        "error": None
                    }
                ],
                "total_queries": 1,
                "score_threshold": 0.7
            }
        }


class EmbeddingGenerationResponse(BaseModel):
    """Response schema for embedding generation"""
    knowledge_base_id: int
    total_chunks: int
    embeddings_generated: int
    embeddings_stored: int
    success: bool
    errors: List[str] = []
    reindexed: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 1,
                "total_chunks": 150,
                "embeddings_generated": 150,
                "embeddings_stored": 150,
                "success": True,
                "errors": [],
                "reindexed": False
            }
        }


class VectorStatisticsResponse(BaseModel):
    """Response schema for vector statistics"""
    client_id: str
    collection_exists: bool
    total_vectors: int
    indexed_vectors: int
    vector_size: int
    distance_metric: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "client_id": "acme-corp",
                "collection_exists": True,
                "total_vectors": 1500,
                "indexed_vectors": 1500,
                "vector_size": 384,
                "distance_metric": "cosine"
            }
        }


class EmbeddingConfig(BaseModel):
    """Embedding configuration schema"""
    provider: str = Field("sentence_transformers", description="Embedding provider")
    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Model name")
    api_key: Optional[str] = Field(None, description="API key for commercial providers")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size for processing")
    normalize: bool = Field(True, description="Whether to normalize embeddings")
    cache_embeddings: bool = Field(True, description="Whether to cache embeddings")
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "sentence_transformers",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "api_key": None,
                "batch_size": 32,
                "normalize": True,
                "cache_embeddings": True
            }
        }


class VectorCollectionInfo(BaseModel):
    """Vector collection information"""
    name: str
    vector_size: int
    distance_metric: str
    points_count: int
    indexed_vectors_count: int
    
    class Config:
        schema_extra = {
            "example": {
                "name": "client_acme-corp",
                "vector_size": 384,
                "distance_metric": "cosine",
                "points_count": 1500,
                "indexed_vectors_count": 1500
            }
        }


class EmbeddingStatistics(BaseModel):
    """Embedding statistics for a knowledge base"""
    knowledge_base_id: int
    total_chunks: int
    embedded_chunks: int
    embedding_coverage: float
    missing_embeddings: int
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 1,
                "total_chunks": 150,
                "embedded_chunks": 150,
                "embedding_coverage": 100.0,
                "missing_embeddings": 0
            }
        }


class VectorMigrationResult(BaseModel):
    """Result of vector migration operation"""
    knowledge_base_id: int
    total_chunks: int
    migrated_chunks: int
    success: bool
    errors: List[str] = []
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 1,
                "total_chunks": 150,
                "migrated_chunks": 150,
                "success": True,
                "errors": []
            }
        }


class VectorHealthCheck(BaseModel):
    """Vector service health check response"""
    status: str
    vector_db_collections: int
    embedding_service: Dict[str, Any]
    cache_size: int
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "vector_db_collections": 5,
                "embedding_service": {
                    "provider": "sentence_transformers",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384
                },
                "cache_size": 1000,
                "error": None
            }
        }


class EmbeddingProviderInfo(BaseModel):
    """Information about embedding providers"""
    available_providers: List[str]
    recommended_models: Dict[str, List[str]]
    default_provider: str
    default_model: str
    
    class Config:
        schema_extra = {
            "example": {
                "available_providers": ["sentence_transformers", "openai"],
                "recommended_models": {
                    "sentence_transformers": [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/all-mpnet-base-v2"
                    ],
                    "openai": [
                        "text-embedding-ada-002",
                        "text-embedding-3-small"
                    ]
                },
                "default_provider": "sentence_transformers",
                "default_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }


# Additional utility schemas

class VectorSearchFilters(BaseModel):
    """Search filters for vector queries"""
    knowledge_base_id: Optional[int] = None
    document_id: Optional[int] = None
    document_type: Optional[str] = None
    date_range: Optional[Dict[str, datetime]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 1,
                "document_id": 45,
                "document_type": "text/markdown",
                "date_range": {
                    "start": "2023-01-01T00:00:00",
                    "end": "2023-12-31T23:59:59"
                },
                "metadata_filters": {
                    "section": "installation",
                    "difficulty": "beginner"
                }
            }
        }


class VectorIndexingStatus(BaseModel):
    """Status of vector indexing operation"""
    knowledge_base_id: int
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    total_chunks: int
    processed_chunks: int
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 1,
                "status": "processing",
                "progress": 0.75,
                "total_chunks": 150,
                "processed_chunks": 112,
                "estimated_completion": "2023-12-01T15:30:00",
                "error_message": None
            }
        }


class VectorSearchAnalytics(BaseModel):
    """Analytics for vector search operations"""
    query: str
    results_count: int
    search_time_ms: int
    top_score: float
    avg_score: float
    embedding_time_ms: int
    vector_search_time_ms: int
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "results_count": 5,
                "search_time_ms": 125,
                "top_score": 0.89,
                "avg_score": 0.76,
                "embedding_time_ms": 45,
                "vector_search_time_ms": 80
            }
        }