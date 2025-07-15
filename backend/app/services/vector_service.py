from typing import List, Dict, Optional, Any, Tuple
import uuid
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.client import DocumentChunk, Document, KnowledgeBase, Client
from ..core.vector_db import VectorDatabase, ClientVectorManager
from ..rag.embeddings import EmbeddingService, EmbeddingConfig
from ..core.database import get_db


class VectorService:
    """Service for managing vector embeddings and search operations"""
    
    def __init__(self, db: Session, embedding_config: EmbeddingConfig = None):
        self.db = db
        self.vector_db = VectorDatabase()
        self.embedding_service = EmbeddingService(embedding_config or EmbeddingConfig())
    
    async def create_client_collection(self, client_id: str, vector_size: int = None) -> bool:
        """Create a vector collection for a client"""
        collection_name = f"client_{client_id}"
        
        # Use embedding service dimension if not specified
        if vector_size is None:
            vector_size = self.embedding_service.get_dimension()
        
        return self.vector_db.create_collection(collection_name, vector_size)
    
    async def delete_client_collection(self, client_id: str) -> bool:
        """Delete a client's vector collection"""
        collection_name = f"client_{client_id}"
        return self.vector_db.delete_collection(collection_name)
    
    async def generate_and_store_embeddings(self, knowledge_base_id: int, force_regenerate: bool = False) -> Dict[str, Any]:
        """Generate embeddings for all chunks in a knowledge base and store in vector DB"""
        
        # Get knowledge base info
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if not kb:
            raise ValueError(f"Knowledge base {knowledge_base_id} not found")
        
        client_id = kb.client_id
        collection_name = f"client_{client_id}"
        
        # Ensure collection exists
        await self.create_client_collection(client_id)
        
        # Get all chunks that need embeddings
        chunks_query = self.db.query(DocumentChunk).join(Document).filter(
            Document.knowledge_base_id == knowledge_base_id
        )
        
        if not force_regenerate:
            # Only get chunks without embeddings
            chunks_query = chunks_query.filter(DocumentChunk.vector_id.is_(None))
        
        chunks = chunks_query.all()
        
        if not chunks:
            return {
                'total_chunks': 0,
                'embeddings_generated': 0,
                'embeddings_stored': 0,
                'errors': []
            }
        
        # Extract texts for embedding generation
        texts = [chunk.chunk_text for chunk in chunks]
        
        # Generate embeddings
        try:
            embedding_results = await self.embedding_service.generate_embeddings(texts)
        except Exception as e:
            return {
                'total_chunks': len(chunks),
                'embeddings_generated': 0,
                'embeddings_stored': 0,
                'errors': [f"Failed to generate embeddings: {str(e)}"]
            }
        
        # Store embeddings in vector database
        points = []
        errors = []
        
        for chunk, embedding_result in zip(chunks, embedding_results):
            try:
                # Generate unique vector ID if not exists
                vector_id = chunk.vector_id or str(uuid.uuid4())
                
                # Prepare point data
                point = {
                    'id': vector_id,
                    'vector': embedding_result.embedding,
                    'payload': {
                        'document_id': chunk.document_id,
                        'chunk_id': chunk.id,
                        'chunk_index': chunk.chunk_index,
                        'knowledge_base_id': knowledge_base_id,
                        'client_id': client_id,
                        'text': chunk.chunk_text,
                        'metadata': chunk.metadata,
                        'embedding_model': embedding_result.model_name,
                        'embedding_provider': embedding_result.provider
                    }
                }
                
                points.append(point)
                
                # Update chunk with vector ID
                chunk.vector_id = vector_id
                
            except Exception as e:
                errors.append(f"Failed to prepare embedding for chunk {chunk.id}: {str(e)}")
        
        # Batch insert into vector database
        embeddings_stored = 0
        if points:
            try:
                success = self.vector_db.upsert_points(collection_name, points)
                if success:
                    embeddings_stored = len(points)
                    # Commit database changes
                    self.db.commit()
                else:
                    errors.append("Failed to store embeddings in vector database")
            except Exception as e:
                errors.append(f"Failed to upsert points: {str(e)}")
        
        return {
            'total_chunks': len(chunks),
            'embeddings_generated': len(embedding_results),
            'embeddings_stored': embeddings_stored,
            'errors': errors
        }
    
    async def search_similar_chunks(
        self, 
        client_id: str, 
        query_text: str, 
        limit: int = 5,
        score_threshold: float = 0.7,
        knowledge_base_id: Optional[int] = None,
        document_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        
        collection_name = f"client_{client_id}"
        
        # Generate query embedding
        try:
            query_embedding_result = await self.embedding_service.generate_embedding(query_text)
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {str(e)}")
        
        # Build filter conditions
        filter_conditions = {'client_id': client_id}
        
        if knowledge_base_id:
            filter_conditions['knowledge_base_id'] = knowledge_base_id
        
        if document_id:
            filter_conditions['document_id'] = document_id
        
        # Search in vector database
        try:
            results = self.vector_db.search_similar(
                collection_name=collection_name,
                query_vector=query_embedding_result.embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions
            )
        except Exception as e:
            raise RuntimeError(f"Failed to search vector database: {str(e)}")
        
        # Enrich results with database information
        enriched_results = []
        for result in results:
            payload = result['payload']
            
            # Get chunk from database for additional info
            chunk = self.db.query(DocumentChunk).filter(
                DocumentChunk.id == payload['chunk_id']
            ).first()
            
            if chunk:
                document = chunk.document
                enriched_result = {
                    'chunk_id': chunk.id,
                    'document_id': document.id,
                    'document_title': document.title,
                    'chunk_index': chunk.chunk_index,
                    'text': chunk.chunk_text,
                    'score': result['score'],
                    'metadata': chunk.metadata,
                    'vector_id': result['id'],
                    'embedding_model': payload.get('embedding_model'),
                    'embedding_provider': payload.get('embedding_provider')
                }
                enriched_results.append(enriched_result)
        
        return enriched_results
    
    async def delete_document_embeddings(self, document_id: int) -> bool:
        """Delete all embeddings for a document"""
        
        # Get document info
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return False
        
        # Get knowledge base and client info
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == document.knowledge_base_id).first()
        if not kb:
            return False
        
        client_id = kb.client_id
        collection_name = f"client_{client_id}"
        
        # Get all chunk vector IDs for this document
        chunks = self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.vector_id.isnot(None)
        ).all()
        
        if not chunks:
            return True  # Nothing to delete
        
        vector_ids = [chunk.vector_id for chunk in chunks]
        
        # Delete from vector database
        try:
            success = self.vector_db.delete_points(collection_name, vector_ids)
            if success:
                # Clear vector IDs from database
                for chunk in chunks:
                    chunk.vector_id = None
                self.db.commit()
            return success
        except Exception as e:
            print(f"Failed to delete embeddings for document {document_id}: {str(e)}")
            return False
    
    async def delete_knowledge_base_embeddings(self, knowledge_base_id: int) -> bool:
        """Delete all embeddings for a knowledge base"""
        
        # Get knowledge base info
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if not kb:
            return False
        
        client_id = kb.client_id
        collection_name = f"client_{client_id}"
        
        # Get all chunk vector IDs for this knowledge base
        chunks = self.db.query(DocumentChunk).join(Document).filter(
            Document.knowledge_base_id == knowledge_base_id,
            DocumentChunk.vector_id.isnot(None)
        ).all()
        
        if not chunks:
            return True  # Nothing to delete
        
        vector_ids = [chunk.vector_id for chunk in chunks]
        
        # Delete from vector database
        try:
            success = self.vector_db.delete_points(collection_name, vector_ids)
            if success:
                # Clear vector IDs from database
                for chunk in chunks:
                    chunk.vector_id = None
                self.db.commit()
            return success
        except Exception as e:
            print(f"Failed to delete embeddings for knowledge base {knowledge_base_id}: {str(e)}")
            return False
    
    async def update_chunk_embedding(self, chunk_id: int, force_regenerate: bool = False) -> bool:
        """Update embedding for a specific chunk"""
        
        chunk = self.db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        if not chunk:
            return False
        
        # Get client info
        document = chunk.document
        kb = document.knowledge_base
        client_id = kb.client_id
        collection_name = f"client_{client_id}"
        
        # Check if embedding already exists
        if chunk.vector_id and not force_regenerate:
            return True  # Already has embedding
        
        # Generate embedding
        try:
            embedding_result = await self.embedding_service.generate_embedding(chunk.chunk_text)
        except Exception as e:
            print(f"Failed to generate embedding for chunk {chunk_id}: {str(e)}")
            return False
        
        # Generate vector ID if not exists
        vector_id = chunk.vector_id or str(uuid.uuid4())
        
        # Prepare point data
        point = {
            'id': vector_id,
            'vector': embedding_result.embedding,
            'payload': {
                'document_id': document.id,
                'chunk_id': chunk.id,
                'chunk_index': chunk.chunk_index,
                'knowledge_base_id': kb.id,
                'client_id': client_id,
                'text': chunk.chunk_text,
                'metadata': chunk.metadata,
                'embedding_model': embedding_result.model_name,
                'embedding_provider': embedding_result.provider
            }
        }
        
        # Store in vector database
        try:
            success = self.vector_db.upsert_points(collection_name, [point])
            if success:
                chunk.vector_id = vector_id
                self.db.commit()
            return success
        except Exception as e:
            print(f"Failed to store embedding for chunk {chunk_id}: {str(e)}")
            return False
    
    def get_vector_statistics(self, client_id: str) -> Dict[str, Any]:
        """Get vector database statistics for a client"""
        
        collection_name = f"client_{client_id}"
        
        # Get collection info
        collection_info = self.vector_db.get_collection_info(collection_name)
        if not collection_info:
            return {
                'collection_exists': False,
                'total_vectors': 0,
                'indexed_vectors': 0,
                'vector_size': 0,
                'distance_metric': None
            }
        
        return {
            'collection_exists': True,
            'total_vectors': collection_info['points_count'],
            'indexed_vectors': collection_info['indexed_vectors_count'],
            'vector_size': collection_info['vector_size'],
            'distance_metric': collection_info['distance']
        }
    
    def get_embedding_statistics(self, knowledge_base_id: int) -> Dict[str, Any]:
        """Get embedding statistics for a knowledge base"""
        
        # Get total chunks and chunks with embeddings
        total_chunks = self.db.query(DocumentChunk).join(Document).filter(
            Document.knowledge_base_id == knowledge_base_id
        ).count()
        
        embedded_chunks = self.db.query(DocumentChunk).join(Document).filter(
            Document.knowledge_base_id == knowledge_base_id,
            DocumentChunk.vector_id.isnot(None)
        ).count()
        
        return {
            'total_chunks': total_chunks,
            'embedded_chunks': embedded_chunks,
            'embedding_coverage': (embedded_chunks / total_chunks * 100) if total_chunks > 0 else 0,
            'missing_embeddings': total_chunks - embedded_chunks
        }
    
    async def reindex_knowledge_base(self, knowledge_base_id: int) -> Dict[str, Any]:
        """Reindex all embeddings for a knowledge base"""
        
        # Delete existing embeddings
        await self.delete_knowledge_base_embeddings(knowledge_base_id)
        
        # Generate new embeddings
        result = await self.generate_and_store_embeddings(knowledge_base_id, force_regenerate=True)
        
        return {
            **result,
            'reindexed': True
        }
    
    async def batch_search(
        self, 
        client_id: str, 
        queries: List[str], 
        limit: int = 5,
        score_threshold: float = 0.7,
        knowledge_base_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Perform batch search for multiple queries"""
        
        results = []
        
        # Process queries concurrently
        tasks = []
        for query in queries:
            task = self.search_similar_chunks(
                client_id=client_id,
                query_text=query,
                limit=limit,
                score_threshold=score_threshold,
                knowledge_base_id=knowledge_base_id
            )
            tasks.append(task)
        
        # Wait for all searches to complete
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        'query': queries[i],
                        'error': str(result),
                        'results': []
                    })
                else:
                    results.append({
                        'query': queries[i],
                        'results': result
                    })
        
        except Exception as e:
            # Fallback to sequential processing
            for query in queries:
                try:
                    search_results = await self.search_similar_chunks(
                        client_id=client_id,
                        query_text=query,
                        limit=limit,
                        score_threshold=score_threshold,
                        knowledge_base_id=knowledge_base_id
                    )
                    results.append({
                        'query': query,
                        'results': search_results
                    })
                except Exception as search_error:
                    results.append({
                        'query': query,
                        'error': str(search_error),
                        'results': []
                    })
        
        return results


class VectorServiceManager:
    """Factory for creating VectorService instances with different embedding configurations"""
    
    @staticmethod
    def create_service(
        db: Session, 
        embedding_provider: str = "sentence_transformers",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None
    ) -> VectorService:
        """Create a VectorService with specified embedding configuration"""
        
        embedding_config = EmbeddingConfig(
            provider=embedding_provider,
            model_name=embedding_model,
            api_key=api_key
        )
        
        return VectorService(db, embedding_config)
    
    @staticmethod
    def create_from_knowledge_base(db: Session, knowledge_base_id: int) -> VectorService:
        """Create a VectorService using embedding config from knowledge base"""
        
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if not kb:
            raise ValueError(f"Knowledge base {knowledge_base_id} not found")
        
        embedding_config_dict = kb.embedding_config or {}
        
        embedding_config = EmbeddingConfig(
            provider=embedding_config_dict.get('provider', 'sentence_transformers'),
            model_name=embedding_config_dict.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
            api_key=embedding_config_dict.get('api_key')
        )
        
        return VectorService(db, embedding_config)


# Utility functions

def get_vector_service(db: Session = None) -> VectorService:
    """Get a default VectorService instance"""
    if db is None:
        db = next(get_db())
    return VectorService(db)


async def ensure_client_vector_setup(client_id: str, db: Session = None) -> bool:
    """Ensure a client has proper vector database setup"""
    if db is None:
        db = next(get_db())
    
    vector_service = VectorService(db)
    return await vector_service.create_client_collection(client_id)


async def migrate_existing_chunks_to_vectors(knowledge_base_id: int, db: Session = None) -> Dict[str, Any]:
    """Migrate existing chunks to vector database (utility for data migration)"""
    if db is None:
        db = next(get_db())
    
    vector_service = VectorService(db)
    return await vector_service.generate_and_store_embeddings(knowledge_base_id)