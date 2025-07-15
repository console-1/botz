from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from ...core.database import get_db
from ...services.hybrid_search import HybridSearchService, HybridSearchConfig, SearchMode, SearchWeights
from ...models.client import Client, KnowledgeBase, Document
from ...schemas.search import (
    HybridSearchRequest, HybridSearchResponse, SearchResultResponse,
    SearchConfigRequest, SearchAnalyticsResponse, QuickSearchRequest
)

router = APIRouter()


@router.post("/clients/{client_id}/search", response_model=HybridSearchResponse)
async def hybrid_search(
    client_id: str,
    search_request: HybridSearchRequest,
    db: Session = Depends(get_db)
):
    """Perform hybrid search combining semantic and keyword approaches"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Optionally verify knowledge base exists
    if search_request.knowledge_base_id:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.id == search_request.knowledge_base_id,
            KnowledgeBase.client_id == client_id
        ).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Optionally verify document exists
    if search_request.document_id:
        doc = db.query(Document).filter(Document.id == search_request.document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
    
    # Create search service
    search_service = HybridSearchService(db)
    
    # Create search configuration
    config = HybridSearchConfig(
        mode=SearchMode(search_request.mode),
        weights=SearchWeights(
            semantic_weight=search_request.semantic_weight,
            keyword_weight=search_request.keyword_weight
        ) if search_request.semantic_weight and search_request.keyword_weight else None,
        semantic_threshold=search_request.semantic_threshold,
        keyword_threshold=search_request.keyword_threshold,
        max_results=search_request.limit,
        enable_reranking=search_request.enable_reranking,
        boost_exact_matches=search_request.boost_exact_matches,
        boost_title_matches=search_request.boost_title_matches,
        boost_recent_documents=search_request.boost_recent_documents
    )
    
    try:
        # Perform search
        results = await search_service.search(
            query=search_request.query,
            client_id=client_id,
            config=config,
            knowledge_base_id=search_request.knowledge_base_id,
            document_id=search_request.document_id
        )
        
        # Convert to response format
        search_results = [
            SearchResultResponse(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_title=result.document_title,
                chunk_index=result.chunk_index,
                text=result.text,
                combined_score=result.combined_score,
                semantic_score=result.semantic_score,
                keyword_score=result.keyword_score,
                metadata=result.metadata,
                vector_id=result.vector_id,
                match_type=result.match_type,
                highlighted_text=result.highlighted_text
            )
            for result in results
        ]
        
        return HybridSearchResponse(
            query=search_request.query,
            results=search_results,
            total_results=len(search_results),
            search_mode=search_request.mode,
            semantic_weight=config.weights.semantic_weight,
            keyword_weight=config.weights.keyword_weight
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/clients/{client_id}/search/analytics", response_model=SearchAnalyticsResponse)
async def hybrid_search_with_analytics(
    client_id: str,
    search_request: HybridSearchRequest,
    db: Session = Depends(get_db)
):
    """Perform hybrid search with detailed analytics"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Create search service
    search_service = HybridSearchService(db)
    
    # Create search configuration
    config = HybridSearchConfig(
        mode=SearchMode(search_request.mode),
        weights=SearchWeights(
            semantic_weight=search_request.semantic_weight,
            keyword_weight=search_request.keyword_weight
        ) if search_request.semantic_weight and search_request.keyword_weight else None,
        semantic_threshold=search_request.semantic_threshold,
        keyword_threshold=search_request.keyword_threshold,
        max_results=search_request.limit,
        enable_reranking=search_request.enable_reranking
    )
    
    try:
        # Perform search with analytics
        results, analytics = await search_service.search_with_analytics(
            query=search_request.query,
            client_id=client_id,
            config=config,
            knowledge_base_id=search_request.knowledge_base_id,
            document_id=search_request.document_id
        )
        
        # Convert results to response format
        search_results = [
            SearchResultResponse(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_title=result.document_title,
                chunk_index=result.chunk_index,
                text=result.text,
                combined_score=result.combined_score,
                semantic_score=result.semantic_score,
                keyword_score=result.keyword_score,
                metadata=result.metadata,
                vector_id=result.vector_id,
                match_type=result.match_type,
                highlighted_text=result.highlighted_text
            )
            for result in results
        ]
        
        return SearchAnalyticsResponse(
            query=search_request.query,
            results=search_results,
            total_results=len(search_results),
            search_mode=search_request.mode,
            semantic_weight=config.weights.semantic_weight,
            keyword_weight=config.weights.keyword_weight,
            analytics=analytics
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search with analytics failed: {str(e)}")


@router.post("/clients/{client_id}/search/quick")
async def quick_search(
    client_id: str,
    search_request: QuickSearchRequest,
    db: Session = Depends(get_db)
):
    """Quick search with default configuration"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    try:
        # Use quick search utility
        from ...services.hybrid_search import quick_search
        
        results = await quick_search(
            query=search_request.query,
            client_id=client_id,
            db=db,
            limit=search_request.limit
        )
        
        return {
            "query": search_request.query,
            "results": results,
            "total_results": len(results),
            "search_mode": "hybrid"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick search failed: {str(e)}")


@router.get("/clients/{client_id}/search/suggestions")
async def get_search_suggestions(
    client_id: str,
    query: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(5, ge=1, le=20, description="Number of suggestions"),
    db: Session = Depends(get_db)
):
    """Get search suggestions based on existing content"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    if len(query.strip()) < 2:
        return {"suggestions": []}
    
    try:
        # Get document titles and chunk content that might match
        from ...models.client import DocumentChunk
        
        # Search in document titles
        title_matches = db.query(Document.title).join(KnowledgeBase).filter(
            KnowledgeBase.client_id == client_id,
            Document.title.ilike(f'%{query}%')
        ).distinct().limit(limit).all()
        
        # Search in chunk content for common phrases
        chunk_matches = db.query(DocumentChunk.chunk_text).join(Document).join(KnowledgeBase).filter(
            KnowledgeBase.client_id == client_id,
            DocumentChunk.chunk_text.ilike(f'%{query}%')
        ).limit(limit * 2).all()
        
        suggestions = []
        
        # Add title matches
        for title_row in title_matches:
            suggestions.append({
                "text": title_row.title,
                "type": "title",
                "highlight": query
            })
        
        # Extract phrases from chunk matches
        import re
        for chunk_row in chunk_matches:
            text = chunk_row.chunk_text
            # Find sentences containing the query
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if query.lower() in sentence.lower():
                    # Extract a reasonable phrase around the query
                    words = sentence.split()
                    if len(words) > 3:
                        suggestions.append({
                            "text": sentence.strip()[:100] + "..." if len(sentence) > 100 else sentence.strip(),
                            "type": "phrase",
                            "highlight": query
                        })
                    break
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion["text"] not in seen:
                seen.add(suggestion["text"])
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= limit:
                    break
        
        return {"suggestions": unique_suggestions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/clients/{client_id}/search/popular-queries")
async def get_popular_queries(
    client_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of popular queries"),
    db: Session = Depends(get_db)
):
    """Get popular search queries for a client (would need query logging)"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # For now, return common/sample queries based on content
    # In a real implementation, this would come from query logs
    
    try:
        # Get some sample queries based on document titles and content
        sample_queries = []
        
        # Get document titles as potential queries
        titles = db.query(Document.title).join(KnowledgeBase).filter(
            KnowledgeBase.client_id == client_id
        ).limit(limit).all()
        
        for title_row in titles:
            # Convert title to question format
            title = title_row.title
            if "how to" not in title.lower():
                sample_queries.append({
                    "query": f"How to {title.lower()}",
                    "frequency": 1,  # Mock frequency
                    "last_searched": "2024-01-01T00:00:00Z"
                })
        
        return {
            "popular_queries": sample_queries[:limit],
            "note": "This is mock data. Implement query logging for real popular queries."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get popular queries: {str(e)}")


@router.get("/search/modes")
async def get_search_modes():
    """Get available search modes and their descriptions"""
    
    return {
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
            },
            {
                "value": "auto",
                "name": "Auto",
                "description": "Automatically selects the best mode based on query characteristics"
            }
        ],
        "default_mode": "hybrid",
        "default_weights": {
            "semantic_weight": 0.7,
            "keyword_weight": 0.3
        }
    }


@router.get("/search/config/recommendations")
async def get_search_config_recommendations(
    query: str = Query(..., description="Query to analyze for recommendations"),
    client_id: str = Query(..., description="Client ID for context")
):
    """Get search configuration recommendations based on query analysis"""
    
    # Analyze query characteristics
    words = query.split()
    
    recommendations = {
        "query": query,
        "analysis": {
            "word_count": len(words),
            "has_quotes": '"' in query,
            "has_technical_terms": any(word.isupper() or word.isdigit() for word in words),
            "has_question_words": any(word.lower() in ['how', 'what', 'when', 'where', 'why', 'who'] for word in words),
            "is_short_query": len(words) <= 2
        }
    }
    
    # Recommend search mode
    if recommendations["analysis"]["has_quotes"]:
        recommendations["recommended_mode"] = "keyword_only"
        recommendations["reason"] = "Query contains quotes suggesting exact matching"
    elif recommendations["analysis"]["is_short_query"]:
        recommendations["recommended_mode"] = "semantic_only"
        recommendations["reason"] = "Short queries benefit from semantic understanding"
    elif recommendations["analysis"]["has_technical_terms"]:
        recommendations["recommended_mode"] = "hybrid"
        recommendations["recommended_weights"] = {"semantic_weight": 0.4, "keyword_weight": 0.6}
        recommendations["reason"] = "Technical terms benefit from keyword search with semantic context"
    else:
        recommendations["recommended_mode"] = "hybrid"
        recommendations["recommended_weights"] = {"semantic_weight": 0.7, "keyword_weight": 0.3}
        recommendations["reason"] = "Natural language queries benefit from balanced hybrid approach"
    
    return recommendations


@router.post("/search/test")
async def test_search_configuration(
    config_request: SearchConfigRequest,
    db: Session = Depends(get_db)
):
    """Test search configuration with sample queries"""
    
    # Create search service
    search_service = HybridSearchService(db)
    
    # Create configuration
    config = HybridSearchConfig(
        mode=SearchMode(config_request.mode),
        weights=SearchWeights(
            semantic_weight=config_request.semantic_weight,
            keyword_weight=config_request.keyword_weight
        ),
        semantic_threshold=config_request.semantic_threshold,
        keyword_threshold=config_request.keyword_threshold,
        max_results=5,  # Limited for testing
        enable_reranking=config_request.enable_reranking
    )
    
    # Test with sample queries
    test_queries = [
        "How to install software",
        "Error message troubleshooting",
        "Configuration settings",
        "API documentation",
        "User guide"
    ]
    
    test_results = []
    
    for query in test_queries:
        try:
            results = await search_service.search(
                query=query,
                client_id=config_request.client_id,
                config=config
            )
            
            test_results.append({
                "query": query,
                "result_count": len(results),
                "top_score": results[0].combined_score if results else 0.0,
                "match_types": list(set(r.match_type for r in results))
            })
        
        except Exception as e:
            test_results.append({
                "query": query,
                "error": str(e),
                "result_count": 0,
                "top_score": 0.0,
                "match_types": []
            })
    
    return {
        "configuration": {
            "mode": config_request.mode,
            "semantic_weight": config_request.semantic_weight,
            "keyword_weight": config_request.keyword_weight,
            "semantic_threshold": config_request.semantic_threshold,
            "keyword_threshold": config_request.keyword_threshold,
            "enable_reranking": config_request.enable_reranking
        },
        "test_results": test_results,
        "overall_performance": {
            "avg_results": sum(r.get("result_count", 0) for r in test_results) / len(test_results),
            "avg_top_score": sum(r.get("top_score", 0) for r in test_results) / len(test_results),
            "successful_queries": len([r for r in test_results if "error" not in r])
        }
    }