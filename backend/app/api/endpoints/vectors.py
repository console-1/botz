from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from ...core.database import get_db
from ...services.vector_service import VectorService, VectorServiceManager
from ...models.client import Client, KnowledgeBase, Document
from ...schemas.vector import (
    VectorSearchRequest, VectorSearchResponse, 
    VectorStatisticsResponse, EmbeddingGenerationResponse,
    BatchSearchRequest, BatchSearchResponse
)

router = APIRouter()


@router.post("/clients/{client_id}/collections/create")
async def create_client_collection(
    client_id: str,
    vector_size: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Create a vector collection for a client"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    vector_service = VectorService(db)
    
    try:
        success = await vector_service.create_client_collection(client_id, vector_size)
        
        if success:
            return {"message": f"Collection created successfully for client {client_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create collection")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@router.delete("/clients/{client_id}/collections")
async def delete_client_collection(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Delete a client's vector collection"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    vector_service = VectorService(db)
    
    try:
        success = await vector_service.delete_client_collection(client_id)
        
        if success:
            return {"message": f"Collection deleted successfully for client {client_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete collection")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/embeddings/generate", response_model=EmbeddingGenerationResponse)
async def generate_knowledge_base_embeddings(
    kb_id: int,
    force_regenerate: bool = False,
    db: Session = Depends(get_db)
):
    """Generate embeddings for all chunks in a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Create vector service with knowledge base embedding config
    vector_service = VectorServiceManager.create_from_knowledge_base(db, kb_id)
    
    try:
        result = await vector_service.generate_and_store_embeddings(kb_id, force_regenerate)
        
        return EmbeddingGenerationResponse(
            knowledge_base_id=kb_id,
            total_chunks=result['total_chunks'],
            embeddings_generated=result['embeddings_generated'],
            embeddings_stored=result['embeddings_stored'],
            success=result['embeddings_stored'] > 0,
            errors=result['errors']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@router.post("/knowledge-bases/{kb_id}/embeddings/reindex", response_model=EmbeddingGenerationResponse)
async def reindex_knowledge_base_embeddings(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Reindex all embeddings for a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Create vector service with knowledge base embedding config
    vector_service = VectorServiceManager.create_from_knowledge_base(db, kb_id)
    
    try:
        result = await vector_service.reindex_knowledge_base(kb_id)
        
        return EmbeddingGenerationResponse(
            knowledge_base_id=kb_id,
            total_chunks=result['total_chunks'],
            embeddings_generated=result['embeddings_generated'],
            embeddings_stored=result['embeddings_stored'],
            success=result['embeddings_stored'] > 0,
            errors=result['errors'],
            reindexed=result.get('reindexed', False)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reindexing embeddings: {str(e)}")


@router.post("/clients/{client_id}/search", response_model=VectorSearchResponse)
async def search_client_vectors(
    client_id: str,
    search_request: VectorSearchRequest,
    db: Session = Depends(get_db)
):
    """Search for similar chunks in a client's vector collection"""
    
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
    
    # Create vector service
    vector_service = VectorService(db)
    
    try:
        results = await vector_service.search_similar_chunks(
            client_id=client_id,
            query_text=search_request.query,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
            knowledge_base_id=search_request.knowledge_base_id,
            document_id=search_request.document_id
        )
        
        return VectorSearchResponse(
            query=search_request.query,
            results=results,
            total_results=len(results),
            score_threshold=search_request.score_threshold
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")


@router.post("/clients/{client_id}/search/batch", response_model=BatchSearchResponse)
async def batch_search_client_vectors(
    client_id: str,
    batch_request: BatchSearchRequest,
    db: Session = Depends(get_db)
):
    """Perform batch search for multiple queries"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    if len(batch_request.queries) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 queries per batch")
    
    # Create vector service
    vector_service = VectorService(db)
    
    try:
        results = await vector_service.batch_search(
            client_id=client_id,
            queries=batch_request.queries,
            limit=batch_request.limit,
            score_threshold=batch_request.score_threshold,
            knowledge_base_id=batch_request.knowledge_base_id
        )
        
        return BatchSearchResponse(
            results=results,
            total_queries=len(batch_request.queries),
            score_threshold=batch_request.score_threshold
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing batch search: {str(e)}")


@router.get("/clients/{client_id}/statistics", response_model=VectorStatisticsResponse)
async def get_client_vector_statistics(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Get vector database statistics for a client"""
    
    # Verify client exists
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    vector_service = VectorService(db)
    
    try:
        stats = vector_service.get_vector_statistics(client_id)
        
        return VectorStatisticsResponse(
            client_id=client_id,
            **stats
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/embedding-statistics")
async def get_knowledge_base_embedding_statistics(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Get embedding statistics for a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    vector_service = VectorService(db)
    
    try:
        stats = vector_service.get_embedding_statistics(kb_id)
        
        return {
            "knowledge_base_id": kb_id,
            **stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding statistics: {str(e)}")


@router.delete("/documents/{doc_id}/embeddings")
async def delete_document_embeddings(
    doc_id: int,
    db: Session = Depends(get_db)
):
    """Delete all embeddings for a document"""
    
    # Verify document exists
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    vector_service = VectorService(db)
    
    try:
        success = await vector_service.delete_document_embeddings(doc_id)
        
        if success:
            return {"message": f"Embeddings deleted successfully for document {doc_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete embeddings")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting embeddings: {str(e)}")


@router.delete("/knowledge-bases/{kb_id}/embeddings")
async def delete_knowledge_base_embeddings(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Delete all embeddings for a knowledge base"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    vector_service = VectorService(db)
    
    try:
        success = await vector_service.delete_knowledge_base_embeddings(kb_id)
        
        if success:
            return {"message": f"Embeddings deleted successfully for knowledge base {kb_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete embeddings")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting embeddings: {str(e)}")


@router.put("/chunks/{chunk_id}/embeddings")
async def update_chunk_embedding(
    chunk_id: int,
    force_regenerate: bool = False,
    db: Session = Depends(get_db)
):
    """Update embedding for a specific chunk"""
    
    from ...models.client import DocumentChunk
    
    # Verify chunk exists
    chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    vector_service = VectorService(db)
    
    try:
        success = await vector_service.update_chunk_embedding(chunk_id, force_regenerate)
        
        if success:
            return {"message": f"Embedding updated successfully for chunk {chunk_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update embedding")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating embedding: {str(e)}")


@router.get("/embedding-providers")
async def get_embedding_providers():
    """Get available embedding providers and models"""
    
    from ...rag.embeddings import EmbeddingService
    
    providers = EmbeddingService.get_available_providers()
    models = EmbeddingService.get_recommended_models()
    
    return {
        "available_providers": providers,
        "recommended_models": models,
        "default_provider": "sentence_transformers",
        "default_model": "sentence-transformers/all-MiniLM-L6-v2"
    }


@router.post("/knowledge-bases/{kb_id}/migrate-chunks")
async def migrate_chunks_to_vectors(
    kb_id: int,
    db: Session = Depends(get_db)
):
    """Migrate existing chunks to vector database (utility endpoint)"""
    
    # Verify knowledge base exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    from ...services.vector_service import migrate_existing_chunks_to_vectors
    
    try:
        result = await migrate_existing_chunks_to_vectors(kb_id, db)
        
        return {
            "message": f"Migration completed for knowledge base {kb_id}",
            "knowledge_base_id": kb_id,
            **result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error migrating chunks: {str(e)}")


@router.get("/health")
async def vector_service_health(db: Session = Depends(get_db)):
    """Check health of vector service components"""
    
    vector_service = VectorService(db)
    
    try:
        # Check vector database connection
        collections = vector_service.vector_db.list_collections()
        
        # Check embedding service
        model_info = vector_service.embedding_service.get_model_info()
        
        return {
            "status": "healthy",
            "vector_db_collections": len(collections),
            "embedding_service": {
                "provider": model_info.get('provider'),
                "model": model_info.get('model_name'),
                "dimension": model_info.get('dimension')
            },
            "cache_size": vector_service.embedding_service.get_cache_size()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }