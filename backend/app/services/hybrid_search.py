from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
import re
import math
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text

from ..models.client import Document, DocumentChunk, KnowledgeBase
from ..services.vector_service import VectorService
from ..rag.embeddings import EmbeddingService, EmbeddingConfig


class SearchMode(Enum):
    """Search modes for hybrid search"""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class SearchWeights:
    """Weights for combining semantic and keyword search results"""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    
    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total = self.semantic_weight + self.keyword_weight
        if total > 0:
            self.semantic_weight /= total
            self.keyword_weight /= total


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""
    mode: SearchMode = SearchMode.HYBRID
    weights: SearchWeights = None
    semantic_threshold: float = 0.7
    keyword_threshold: float = 0.1
    max_results: int = 20
    enable_reranking: bool = True
    boost_exact_matches: float = 1.5
    boost_title_matches: float = 1.2
    boost_recent_documents: float = 1.1
    min_query_length: int = 2
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = SearchWeights()


@dataclass
class SearchResult:
    """Individual search result with scoring details"""
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
    match_type: str = "hybrid"  # semantic, keyword, hybrid
    highlighted_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'document_title': self.document_title,
            'chunk_index': self.chunk_index,
            'text': self.text,
            'combined_score': self.combined_score,
            'semantic_score': self.semantic_score,
            'keyword_score': self.keyword_score,
            'metadata': self.metadata,
            'vector_id': self.vector_id,
            'match_type': self.match_type,
            'highlighted_text': self.highlighted_text
        }


class KeywordSearchEngine:
    """Handles keyword-based search functionality"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def search(
        self,
        query: str,
        client_id: str,
        knowledge_base_id: Optional[int] = None,
        document_id: Optional[int] = None,
        limit: int = 20,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using database full-text search"""
        
        # Clean and prepare query
        cleaned_query = self._clean_query(query)
        if not cleaned_query:
            return []
        
        # Build base query
        base_query = self.db.query(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.chunk_text,
            DocumentChunk.metadata,
            DocumentChunk.vector_id,
            Document.title,
            Document.created_at
        ).join(Document).join(KnowledgeBase).filter(
            KnowledgeBase.client_id == client_id
        )
        
        # Apply filters
        if knowledge_base_id:
            base_query = base_query.filter(Document.knowledge_base_id == knowledge_base_id)
        
        if document_id:
            base_query = base_query.filter(Document.id == document_id)
        
        # Perform different types of keyword matching
        results = []
        
        # 1. Exact phrase matching
        exact_results = self._search_exact_phrase(base_query, cleaned_query, limit)
        results.extend(exact_results)
        
        # 2. Multi-word matching (all words must appear)
        if len(cleaned_query.split()) > 1:
            multi_word_results = self._search_multi_word(base_query, cleaned_query, limit)
            results.extend(multi_word_results)
        
        # 3. Individual word matching
        word_results = self._search_individual_words(base_query, cleaned_query, limit)
        results.extend(word_results)
        
        # 4. Fuzzy matching for typos
        fuzzy_results = self._search_fuzzy(base_query, cleaned_query, limit)
        results.extend(fuzzy_results)
        
        # Remove duplicates and score results
        unique_results = self._deduplicate_and_score(results, cleaned_query, threshold)
        
        # Sort by score and limit results
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results[:limit]
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query string"""
        # Remove special characters, normalize whitespace
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _search_exact_phrase(self, base_query, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for exact phrase matches"""
        results = []
        
        # Use ILIKE for case-insensitive exact phrase matching
        phrase_query = base_query.filter(
            DocumentChunk.chunk_text.ilike(f'%{query}%')
        ).limit(limit).all()
        
        for row in phrase_query:
            results.append({
                'chunk_id': row.id,
                'document_id': row.document_id,
                'chunk_index': row.chunk_index,
                'text': row.chunk_text,
                'metadata': row.metadata,
                'vector_id': row.vector_id,
                'document_title': row.title,
                'created_at': row.created_at,
                'match_type': 'exact_phrase',
                'match_strength': 1.0
            })
        
        return results
    
    def _search_multi_word(self, base_query, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for chunks containing all words"""
        words = query.split()
        if len(words) < 2:
            return []
        
        results = []
        
        # Build filter for all words
        filters = []
        for word in words:
            filters.append(DocumentChunk.chunk_text.ilike(f'%{word}%'))
        
        multi_word_query = base_query.filter(and_(*filters)).limit(limit).all()
        
        for row in multi_word_query:
            results.append({
                'chunk_id': row.id,
                'document_id': row.document_id,
                'chunk_index': row.chunk_index,
                'text': row.chunk_text,
                'metadata': row.metadata,
                'vector_id': row.vector_id,
                'document_title': row.title,
                'created_at': row.created_at,
                'match_type': 'multi_word',
                'match_strength': 0.8
            })
        
        return results
    
    def _search_individual_words(self, base_query, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for chunks containing any of the words"""
        words = query.split()
        results = []
        
        # Build filter for any word
        filters = []
        for word in words:
            filters.append(DocumentChunk.chunk_text.ilike(f'%{word}%'))
        
        word_query = base_query.filter(or_(*filters)).limit(limit).all()
        
        for row in word_query:
            # Calculate match strength based on word coverage
            text_lower = row.chunk_text.lower()
            matching_words = sum(1 for word in words if word in text_lower)
            match_strength = matching_words / len(words) * 0.6
            
            results.append({
                'chunk_id': row.id,
                'document_id': row.document_id,
                'chunk_index': row.chunk_index,
                'text': row.chunk_text,
                'metadata': row.metadata,
                'vector_id': row.vector_id,
                'document_title': row.title,
                'created_at': row.created_at,
                'match_type': 'individual_words',
                'match_strength': match_strength
            })
        
        return results
    
    def _search_fuzzy(self, base_query, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search with fuzzy matching for typos"""
        results = []
        
        # Use PostgreSQL's similarity function if available
        try:
            # This requires pg_trgm extension
            fuzzy_query = base_query.filter(
                func.similarity(DocumentChunk.chunk_text, query) > 0.3
            ).limit(limit).all()
            
            for row in fuzzy_query:
                results.append({
                    'chunk_id': row.id,
                    'document_id': row.document_id,
                    'chunk_index': row.chunk_index,
                    'text': row.chunk_text,
                    'metadata': row.metadata,
                    'vector_id': row.vector_id,
                    'document_title': row.title,
                    'created_at': row.created_at,
                    'match_type': 'fuzzy',
                    'match_strength': 0.4
                })
        except Exception:
            # Fallback to simple partial matching
            words = query.split()
            for word in words:
                if len(word) > 3:  # Only for longer words
                    partial_query = base_query.filter(
                        DocumentChunk.chunk_text.ilike(f'%{word[:-1]}%')
                    ).limit(limit // len(words)).all()
                    
                    for row in partial_query:
                        results.append({
                            'chunk_id': row.id,
                            'document_id': row.document_id,
                            'chunk_index': row.chunk_index,
                            'text': row.chunk_text,
                            'metadata': row.metadata,
                            'vector_id': row.vector_id,
                            'document_title': row.title,
                            'created_at': row.created_at,
                            'match_type': 'partial',
                            'match_strength': 0.3
                        })
        
        return results
    
    def _deduplicate_and_score(self, results: List[Dict[str, Any]], query: str, threshold: float) -> List[Dict[str, Any]]:
        """Remove duplicates and calculate keyword scores"""
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_id = result['chunk_id']
            if chunk_id in seen_chunks:
                continue
            
            seen_chunks.add(chunk_id)
            
            # Calculate comprehensive keyword score
            score = self._calculate_keyword_score(result, query)
            
            if score >= threshold:
                result['score'] = score
                unique_results.append(result)
        
        return unique_results
    
    def _calculate_keyword_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate keyword matching score"""
        text = result['text'].lower()
        title = result['document_title'].lower()
        query_lower = query.lower()
        words = query_lower.split()
        
        score = 0.0
        
        # Base score from match type
        match_type_scores = {
            'exact_phrase': 1.0,
            'multi_word': 0.8,
            'individual_words': 0.6,
            'fuzzy': 0.4,
            'partial': 0.3
        }
        
        base_score = match_type_scores.get(result['match_type'], 0.5)
        score += base_score * 0.4
        
        # Exact phrase bonus
        if query_lower in text:
            score += 0.3
        
        # Title matching bonus
        if any(word in title for word in words):
            score += 0.2
        
        # Word frequency scoring
        word_scores = []
        for word in words:
            if word in text:
                # TF-IDF-like scoring
                tf = text.count(word) / len(text.split())
                word_scores.append(tf)
        
        if word_scores:
            score += sum(word_scores) / len(word_scores) * 0.1
        
        # Position bonus (earlier matches are better)
        first_match_pos = text.find(query_lower)
        if first_match_pos != -1:
            position_bonus = max(0, 1 - first_match_pos / len(text)) * 0.1
            score += position_bonus
        
        return min(score, 1.0)
    
    def highlight_matches(self, text: str, query: str, max_length: int = 200) -> str:
        """Highlight matching terms in text"""
        words = query.lower().split()
        highlighted = text
        
        # Highlight each word
        for word in words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(f'<mark>{word}</mark>', highlighted)
        
        # Truncate if needed, preserving highlights
        if len(highlighted) > max_length:
            # Try to center around first highlight
            mark_pos = highlighted.find('<mark>')
            if mark_pos != -1:
                start = max(0, mark_pos - max_length // 2)
                end = start + max_length
                highlighted = highlighted[start:end]
                if start > 0:
                    highlighted = '...' + highlighted
                if end < len(text):
                    highlighted = highlighted + '...'
        
        return highlighted


class HybridSearchService:
    """Main hybrid search service combining semantic and keyword search"""
    
    def __init__(self, db: Session, embedding_config: EmbeddingConfig = None):
        self.db = db
        self.vector_service = VectorService(db, embedding_config)
        self.keyword_engine = KeywordSearchEngine(db)
        self.embedding_service = EmbeddingService(embedding_config or EmbeddingConfig())
    
    async def search(
        self,
        query: str,
        client_id: str,
        config: HybridSearchConfig = None,
        knowledge_base_id: Optional[int] = None,
        document_id: Optional[int] = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword approaches"""
        
        if config is None:
            config = HybridSearchConfig()
        
        # Validate query
        if not query or len(query.strip()) < config.min_query_length:
            return []
        
        query = query.strip()
        
        # Determine search mode
        search_mode = self._determine_search_mode(query, config)
        
        # Perform searches based on mode
        semantic_results = []
        keyword_results = []
        
        if search_mode in [SearchMode.SEMANTIC_ONLY, SearchMode.HYBRID]:
            semantic_results = await self._perform_semantic_search(
                query, client_id, knowledge_base_id, document_id, config
            )
        
        if search_mode in [SearchMode.KEYWORD_ONLY, SearchMode.HYBRID]:
            keyword_results = self._perform_keyword_search(
                query, client_id, knowledge_base_id, document_id, config
            )
        
        # Combine and rank results
        combined_results = self._combine_results(
            semantic_results, keyword_results, config, query
        )
        
        # Apply reranking if enabled
        if config.enable_reranking:
            combined_results = self._rerank_results(combined_results, query, config)
        
        # Apply final filtering and sorting
        final_results = self._finalize_results(combined_results, config, query)
        
        return final_results[:config.max_results]
    
    def _determine_search_mode(self, query: str, config: HybridSearchConfig) -> SearchMode:
        """Determine the best search mode for the query"""
        
        if config.mode != SearchMode.AUTO:
            return config.mode
        
        # Auto-determine based on query characteristics
        words = query.split()
        
        # Short queries with common words might benefit from semantic search
        if len(words) <= 2:
            return SearchMode.SEMANTIC_ONLY
        
        # Queries with quotes suggest exact matching
        if '"' in query:
            return SearchMode.KEYWORD_ONLY
        
        # Technical terms or specific names benefit from keyword search
        if any(word.isupper() or word.isdigit() for word in words):
            return SearchMode.HYBRID
        
        # Default to hybrid for most queries
        return SearchMode.HYBRID
    
    async def _perform_semantic_search(
        self,
        query: str,
        client_id: str,
        knowledge_base_id: Optional[int],
        document_id: Optional[int],
        config: HybridSearchConfig
    ) -> List[Dict[str, Any]]:
        """Perform semantic vector search"""
        
        try:
            results = await self.vector_service.search_similar_chunks(
                client_id=client_id,
                query_text=query,
                limit=config.max_results * 2,  # Get more to allow for deduplication
                score_threshold=config.semantic_threshold,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id
            )
            
            # Convert to standard format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'chunk_id': result['chunk_id'],
                    'document_id': result['document_id'],
                    'chunk_index': result['chunk_index'],
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'vector_id': result['vector_id'],
                    'document_title': result['document_title'],
                    'score': result['score'],
                    'match_type': 'semantic'
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Semantic search failed: {str(e)}")
            return []
    
    def _perform_keyword_search(
        self,
        query: str,
        client_id: str,
        knowledge_base_id: Optional[int],
        document_id: Optional[int],
        config: HybridSearchConfig
    ) -> List[Dict[str, Any]]:
        """Perform keyword search"""
        
        try:
            results = self.keyword_engine.search(
                query=query,
                client_id=client_id,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                limit=config.max_results * 2,
                threshold=config.keyword_threshold
            )
            
            # Add match type
            for result in results:
                result['match_type'] = 'keyword'
            
            return results
            
        except Exception as e:
            print(f"Keyword search failed: {str(e)}")
            return []
    
    def _combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        config: HybridSearchConfig,
        query: str
    ) -> List[SearchResult]:
        """Combine semantic and keyword results"""
        
        # Index results by chunk_id for deduplication
        combined_by_chunk = {}
        
        # Process semantic results
        for result in semantic_results:
            chunk_id = result['chunk_id']
            combined_by_chunk[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                document_id=result['document_id'],
                document_title=result['document_title'],
                chunk_index=result['chunk_index'],
                text=result['text'],
                combined_score=result['score'] * config.weights.semantic_weight,
                semantic_score=result['score'],
                keyword_score=0.0,
                metadata=result['metadata'],
                vector_id=result.get('vector_id'),
                match_type='semantic'
            )
        
        # Process keyword results
        for result in keyword_results:
            chunk_id = result['chunk_id']
            keyword_score = result['score']
            
            if chunk_id in combined_by_chunk:
                # Combine with existing semantic result
                existing = combined_by_chunk[chunk_id]
                existing.keyword_score = keyword_score
                existing.combined_score = (
                    existing.semantic_score * config.weights.semantic_weight +
                    keyword_score * config.weights.keyword_weight
                )
                existing.match_type = 'hybrid'
            else:
                # Create new keyword-only result
                combined_by_chunk[chunk_id] = SearchResult(
                    chunk_id=chunk_id,
                    document_id=result['document_id'],
                    document_title=result['document_title'],
                    chunk_index=result['chunk_index'],
                    text=result['text'],
                    combined_score=keyword_score * config.weights.keyword_weight,
                    semantic_score=0.0,
                    keyword_score=keyword_score,
                    metadata=result['metadata'],
                    vector_id=result.get('vector_id'),
                    match_type='keyword'
                )
        
        # Apply boosts
        for result in combined_by_chunk.values():
            result.combined_score = self._apply_boosts(result, query, config)
        
        return list(combined_by_chunk.values())
    
    def _apply_boosts(self, result: SearchResult, query: str, config: HybridSearchConfig) -> float:
        """Apply scoring boosts based on various factors"""
        
        score = result.combined_score
        
        # Exact match boost
        if query.lower() in result.text.lower():
            score *= config.boost_exact_matches
        
        # Title match boost
        if any(word.lower() in result.document_title.lower() for word in query.split()):
            score *= config.boost_title_matches
        
        # Recent document boost (if we have created_at in metadata)
        if 'created_at' in result.metadata:
            # Simple recency boost - could be more sophisticated
            score *= config.boost_recent_documents
        
        # Chunk position boost (earlier chunks often more important)
        if result.chunk_index == 0:
            score *= 1.1
        
        return score
    
    def _rerank_results(self, results: List[SearchResult], query: str, config: HybridSearchConfig) -> List[SearchResult]:
        """Apply reranking to improve result quality"""
        
        # Simple reranking based on query-result similarity
        query_words = set(query.lower().split())
        
        for result in results:
            # Calculate additional relevance factors
            text_words = set(result.text.lower().split())
            
            # Word overlap boost
            overlap = len(query_words.intersection(text_words))
            if overlap > 0:
                overlap_boost = overlap / len(query_words) * 0.1
                result.combined_score += overlap_boost
            
            # Length normalization (prefer more substantial chunks)
            text_length = len(result.text.split())
            if text_length > 50:  # Substantial content
                result.combined_score *= 1.05
            elif text_length < 10:  # Very short content
                result.combined_score *= 0.95
        
        return results
    
    def _finalize_results(self, results: List[SearchResult], config: HybridSearchConfig, query: str = "") -> List[SearchResult]:
        """Final processing and sorting of results"""
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Add highlighting
        for result in results:
            if result.match_type in ['keyword', 'hybrid'] and query:
                result.highlighted_text = self.keyword_engine.highlight_matches(
                    result.text, query
                )
        
        return results
    
    async def search_with_analytics(
        self,
        query: str,
        client_id: str,
        config: HybridSearchConfig = None,
        knowledge_base_id: Optional[int] = None,
        document_id: Optional[int] = None
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Perform search with detailed analytics"""
        
        import time
        start_time = time.time()
        
        # Perform search
        results = await self.search(query, client_id, config, knowledge_base_id, document_id)
        
        end_time = time.time()
        
        # Calculate analytics
        analytics = {
            'query': query,
            'total_results': len(results),
            'search_time_ms': int((end_time - start_time) * 1000),
            'mode_used': config.mode.value if config else SearchMode.HYBRID.value,
            'semantic_results': len([r for r in results if r.match_type in ['semantic', 'hybrid']]),
            'keyword_results': len([r for r in results if r.match_type in ['keyword', 'hybrid']]),
            'hybrid_results': len([r for r in results if r.match_type == 'hybrid']),
            'top_score': results[0].combined_score if results else 0.0,
            'avg_score': sum(r.combined_score for r in results) / len(results) if results else 0.0,
            'score_distribution': self._analyze_score_distribution(results)
        }
        
        return results, analytics
    
    def _analyze_score_distribution(self, results: List[SearchResult]) -> Dict[str, int]:
        """Analyze the distribution of scores"""
        
        if not results:
            return {}
        
        distribution = {
            'excellent': 0,  # > 0.9
            'good': 0,       # 0.7 - 0.9
            'fair': 0,       # 0.5 - 0.7
            'poor': 0        # < 0.5
        }
        
        for result in results:
            score = result.combined_score
            if score > 0.9:
                distribution['excellent'] += 1
            elif score > 0.7:
                distribution['good'] += 1
            elif score > 0.5:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution


# Utility functions

def create_hybrid_search_service(db: Session, embedding_config: EmbeddingConfig = None) -> HybridSearchService:
    """Create a hybrid search service with default configuration"""
    return HybridSearchService(db, embedding_config)


def create_search_config(
    mode: SearchMode = SearchMode.HYBRID,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    **kwargs
) -> HybridSearchConfig:
    """Create search configuration with custom weights"""
    weights = SearchWeights(semantic_weight, keyword_weight)
    return HybridSearchConfig(mode=mode, weights=weights, **kwargs)


async def quick_search(
    query: str,
    client_id: str,
    db: Session,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Quick search with default configuration"""
    service = HybridSearchService(db)
    config = HybridSearchConfig(max_results=limit)
    
    results = await service.search(query, client_id, config)
    return [result.to_dict() for result in results]