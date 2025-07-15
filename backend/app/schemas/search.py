from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class SearchModeEnum(str, Enum):
    """Search mode options"""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    AUTO = "auto"


class HybridSearchRequest(BaseModel):
    """Request schema for hybrid search"""
    query: str = Field(..., description="Search query text")
    mode: SearchModeEnum = Field(SearchModeEnum.HYBRID, description="Search mode")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    
    # Weight configuration
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for semantic search")
    keyword_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for keyword search")
    
    # Thresholds
    semantic_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum semantic similarity score")
    keyword_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum keyword matching score")
    
    # Filters
    knowledge_base_id: Optional[int] = Field(None, description="Filter by knowledge base ID")
    document_id: Optional[int] = Field(None, description="Filter by document ID")
    
    # Enhancement options
    enable_reranking: bool = Field(True, description="Enable result reranking")
    boost_exact_matches: float = Field(1.5, ge=1.0, le=3.0, description="Boost factor for exact matches")
    boost_title_matches: float = Field(1.2, ge=1.0, le=3.0, description="Boost factor for title matches")
    boost_recent_documents: float = Field(1.1, ge=1.0, le=3.0, description="Boost factor for recent documents")
    
    @validator('semantic_weight', 'keyword_weight')
    def validate_weights(cls, v, values):
        if v is not None:
            # If one weight is provided, the other should be too
            semantic_weight = values.get('semantic_weight')
            keyword_weight = values.get('keyword_weight')
            
            if semantic_weight is not None and keyword_weight is not None:
                if abs(semantic_weight + keyword_weight - 1.0) > 0.001:
                    raise ValueError('semantic_weight and keyword_weight must sum to 1.0')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "mode": "hybrid",
                "limit": 10,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "semantic_threshold": 0.7,
                "keyword_threshold": 0.1,
                "knowledge_base_id": 1,
                "enable_reranking": True,
                "boost_exact_matches": 1.5,
                "boost_title_matches": 1.2,
                "boost_recent_documents": 1.1
            }
        }


class SearchResultResponse(BaseModel):
    """Individual search result response"""
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    combined_score: float
    semantic_score: float
    keyword_score: float
    metadata: Dict[str, Any]
    vector_id: Optional[str] = None
    match_type: str  # semantic, keyword, hybrid
    highlighted_text: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "chunk_id": 123,
                "document_id": 45,
                "document_title": "Installation Guide",
                "chunk_index": 2,
                "text": "To install the software, follow these steps...",
                "combined_score": 0.85,
                "semantic_score": 0.82,
                "keyword_score": 0.91,
                "metadata": {"section": "installation", "difficulty": "beginner"},
                "vector_id": "uuid-123",
                "match_type": "hybrid",
                "highlighted_text": "To install the <mark>software</mark>, follow these steps..."
            }
        }


class HybridSearchResponse(BaseModel):
    """Response schema for hybrid search"""
    query: str
    results: List[SearchResultResponse]
    total_results: int
    search_mode: SearchModeEnum
    semantic_weight: float
    keyword_weight: float
    
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
                        "combined_score": 0.85,
                        "semantic_score": 0.82,
                        "keyword_score": 0.91,
                        "metadata": {"section": "installation"},
                        "vector_id": "uuid-123",
                        "match_type": "hybrid",
                        "highlighted_text": "To install the <mark>software</mark>, follow these steps..."
                    }
                ],
                "total_results": 1,
                "search_mode": "hybrid",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3
            }
        }


class SearchAnalyticsResponse(BaseModel):
    """Response schema for search with analytics"""
    query: str
    results: List[SearchResultResponse]
    total_results: int
    search_mode: SearchModeEnum
    semantic_weight: float
    keyword_weight: float
    analytics: Dict[str, Any]
    
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
                        "combined_score": 0.85,
                        "semantic_score": 0.82,
                        "keyword_score": 0.91,
                        "metadata": {"section": "installation"},
                        "vector_id": "uuid-123",
                        "match_type": "hybrid"
                    }
                ],
                "total_results": 1,
                "search_mode": "hybrid",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "analytics": {
                    "search_time_ms": 125,
                    "semantic_results": 1,
                    "keyword_results": 1,
                    "hybrid_results": 1,
                    "top_score": 0.85,
                    "avg_score": 0.85,
                    "score_distribution": {
                        "excellent": 1,
                        "good": 0,
                        "fair": 0,
                        "poor": 0
                    }
                }
            }
        }


class QuickSearchRequest(BaseModel):
    """Request schema for quick search"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "limit": 10
            }
        }


class SearchConfigRequest(BaseModel):
    """Request schema for testing search configuration"""
    client_id: str = Field(..., description="Client ID for testing")
    mode: SearchModeEnum = Field(SearchModeEnum.HYBRID, description="Search mode")
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for semantic search")
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for keyword search")
    semantic_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum semantic similarity score")
    keyword_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum keyword matching score")
    enable_reranking: bool = Field(True, description="Enable result reranking")
    
    @validator('keyword_weight')
    def validate_weights_sum(cls, v, values):
        semantic_weight = values.get('semantic_weight', 0.7)
        if abs(semantic_weight + v - 1.0) > 0.001:
            raise ValueError('semantic_weight and keyword_weight must sum to 1.0')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "client_id": "acme-corp",
                "mode": "hybrid",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "semantic_threshold": 0.7,
                "keyword_threshold": 0.1,
                "enable_reranking": True
            }
        }


class SearchSuggestion(BaseModel):
    """Individual search suggestion"""
    text: str
    type: str  # title, phrase, entity
    highlight: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "How to install the software",
                "type": "title",
                "highlight": "install"
            }
        }


class SearchSuggestionsResponse(BaseModel):
    """Response schema for search suggestions"""
    suggestions: List[SearchSuggestion]
    
    class Config:
        schema_extra = {
            "example": {
                "suggestions": [
                    {
                        "text": "How to install the software",
                        "type": "title",
                        "highlight": "install"
                    },
                    {
                        "text": "Installation troubleshooting guide",
                        "type": "phrase",
                        "highlight": "install"
                    }
                ]
            }
        }


class PopularQuery(BaseModel):
    """Popular query item"""
    query: str
    frequency: int
    last_searched: str
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "frequency": 25,
                "last_searched": "2024-01-15T10:30:00Z"
            }
        }


class PopularQueriesResponse(BaseModel):
    """Response schema for popular queries"""
    popular_queries: List[PopularQuery]
    note: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "popular_queries": [
                    {
                        "query": "How to install the software?",
                        "frequency": 25,
                        "last_searched": "2024-01-15T10:30:00Z"
                    },
                    {
                        "query": "Troubleshooting connection issues",
                        "frequency": 18,
                        "last_searched": "2024-01-14T14:20:00Z"
                    }
                ]
            }
        }


class SearchMode(BaseModel):
    """Search mode information"""
    value: str
    name: str
    description: str
    
    class Config:
        schema_extra = {
            "example": {
                "value": "hybrid",
                "name": "Hybrid",
                "description": "Combines semantic and keyword search with configurable weights"
            }
        }


class SearchModesResponse(BaseModel):
    """Response schema for available search modes"""
    modes: List[SearchMode]
    default_mode: str
    default_weights: Dict[str, float]
    
    class Config:
        schema_extra = {
            "example": {
                "modes": [
                    {
                        "value": "semantic_only",
                        "name": "Semantic Only",
                        "description": "Uses only vector similarity search for conceptual matching"
                    },
                    {
                        "value": "keyword_only",
                        "name": "Keyword Only",
                        "description": "Uses only keyword-based search for exact term matching"
                    },
                    {
                        "value": "hybrid",
                        "name": "Hybrid",
                        "description": "Combines semantic and keyword search with configurable weights"
                    }
                ],
                "default_mode": "hybrid",
                "default_weights": {
                    "semantic_weight": 0.7,
                    "keyword_weight": 0.3
                }
            }
        }


class QueryAnalysis(BaseModel):
    """Query analysis results"""
    word_count: int
    has_quotes: bool
    has_technical_terms: bool
    has_question_words: bool
    is_short_query: bool
    
    class Config:
        schema_extra = {
            "example": {
                "word_count": 5,
                "has_quotes": False,
                "has_technical_terms": False,
                "has_question_words": True,
                "is_short_query": False
            }
        }


class SearchConfigRecommendation(BaseModel):
    """Search configuration recommendation"""
    query: str
    analysis: QueryAnalysis
    recommended_mode: SearchModeEnum
    recommended_weights: Optional[Dict[str, float]] = None
    reason: str
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install the software?",
                "analysis": {
                    "word_count": 5,
                    "has_quotes": False,
                    "has_technical_terms": False,
                    "has_question_words": True,
                    "is_short_query": False
                },
                "recommended_mode": "hybrid",
                "recommended_weights": {
                    "semantic_weight": 0.7,
                    "keyword_weight": 0.3
                },
                "reason": "Natural language queries benefit from balanced hybrid approach"
            }
        }


class SearchTestResult(BaseModel):
    """Individual search test result"""
    query: str
    result_count: int
    top_score: float
    match_types: List[str]
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How to install software",
                "result_count": 5,
                "top_score": 0.89,
                "match_types": ["hybrid", "semantic"],
                "error": None
            }
        }


class SearchTestPerformance(BaseModel):
    """Overall search test performance"""
    avg_results: float
    avg_top_score: float
    successful_queries: int
    
    class Config:
        schema_extra = {
            "example": {
                "avg_results": 4.2,
                "avg_top_score": 0.78,
                "successful_queries": 5
            }
        }


class SearchTestResponse(BaseModel):
    """Response schema for search configuration testing"""
    configuration: Dict[str, Any]
    test_results: List[SearchTestResult]
    overall_performance: SearchTestPerformance
    
    class Config:
        schema_extra = {
            "example": {
                "configuration": {
                    "mode": "hybrid",
                    "semantic_weight": 0.7,
                    "keyword_weight": 0.3,
                    "semantic_threshold": 0.7,
                    "keyword_threshold": 0.1,
                    "enable_reranking": True
                },
                "test_results": [
                    {
                        "query": "How to install software",
                        "result_count": 5,
                        "top_score": 0.89,
                        "match_types": ["hybrid", "semantic"]
                    }
                ],
                "overall_performance": {
                    "avg_results": 4.2,
                    "avg_top_score": 0.78,
                    "successful_queries": 5
                }
            }
        }


# Additional utility schemas

class SearchFilter(BaseModel):
    """Search filter options"""
    knowledge_base_id: Optional[int] = None
    document_id: Optional[int] = None
    document_type: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "knowledge_base_id": 1,
                "document_id": 45,
                "document_type": "text/markdown",
                "date_range": {
                    "start": "2024-01-01",
                    "end": "2024-12-31"
                },
                "metadata_filters": {
                    "section": "installation",
                    "difficulty": "beginner"
                }
            }
        }


class SearchPerformanceMetrics(BaseModel):
    """Search performance metrics"""
    query_time_ms: int
    semantic_search_time_ms: int
    keyword_search_time_ms: int
    reranking_time_ms: int
    total_chunks_searched: int
    cache_hits: int
    
    class Config:
        schema_extra = {
            "example": {
                "query_time_ms": 125,
                "semantic_search_time_ms": 80,
                "keyword_search_time_ms": 30,
                "reranking_time_ms": 15,
                "total_chunks_searched": 1500,
                "cache_hits": 3
            }
        }


class SearchQualityMetrics(BaseModel):
    """Search quality metrics"""
    precision_at_k: Dict[int, float]  # Precision at k (1, 3, 5, 10)
    recall_at_k: Dict[int, float]     # Recall at k
    mrr: float                        # Mean Reciprocal Rank
    ndcg: float                       # Normalized Discounted Cumulative Gain
    
    class Config:
        schema_extra = {
            "example": {
                "precision_at_k": {1: 0.95, 3: 0.87, 5: 0.82, 10: 0.78},
                "recall_at_k": {1: 0.15, 3: 0.38, 5: 0.52, 10: 0.68},
                "mrr": 0.85,
                "ndcg": 0.82
            }
        }