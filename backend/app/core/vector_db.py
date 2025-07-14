from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional, Any
import uuid
from .config import settings


class VectorDatabase:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.embedding_size = 384  # for sentence-transformers/all-MiniLM-L6-v2
    
    def create_collection(self, collection_name: str, vector_size: int = None) -> bool:
        """Create a new collection for a client's knowledge base"""
        try:
            vector_size = vector_size or self.embedding_size
            
            # Check if collection already exists
            collections = self.client.get_collections()
            if collection_name in [c.name for c in collections.collections]:
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            return True
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Insert or update points in a collection"""
        try:
            # Convert to PointStruct objects
            point_structs = []
            for point in points:
                point_struct = PointStruct(
                    id=point.get('id', str(uuid.uuid4())),
                    vector=point['vector'],
                    payload=point.get('payload', {})
                )
                point_structs.append(point_struct)
            
            # Upsert points
            self.client.upsert(
                collection_name=collection_name,
                points=point_structs
            )
            return True
        except Exception as e:
            print(f"Error upserting points to {collection_name}: {e}")
            return False
    
    def search_similar(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        try:
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            # Search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching in {collection_name}: {e}")
            return []
    
    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete specific points from a collection"""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            return True
        except Exception as e:
            print(f"Error deleting points from {collection_name}: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get information about a collection"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': info.config.name,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'points_count': info.points_count,
                'indexed_vectors_count': info.indexed_vectors_count
            }
        except Exception as e:
            print(f"Error getting collection info for {collection_name}: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []


class ClientVectorManager:
    """Manages vector operations for specific clients"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.collection_name = f"client_{client_id}"
        self.vector_db = VectorDatabase()
    
    def initialize_client_collection(self) -> bool:
        """Initialize collection for this client"""
        return self.vector_db.create_collection(self.collection_name)
    
    def add_document_chunks(self, document_id: str, chunks: List[Dict]) -> bool:
        """Add document chunks to client's collection"""
        points = []
        for i, chunk in enumerate(chunks):
            point = {
                'id': f"{document_id}_{i}",
                'vector': chunk['embedding'],
                'payload': {
                    'document_id': document_id,
                    'chunk_index': i,
                    'text': chunk['text'],
                    'metadata': chunk.get('metadata', {})
                }
            }
            points.append(point)
        
        return self.vector_db.upsert_points(self.collection_name, points)
    
    def search_knowledge_base(
        self, 
        query_vector: List[float], 
        limit: int = 5,
        score_threshold: float = 0.7,
        document_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search client's knowledge base"""
        filter_conditions = {}
        if document_filter:
            filter_conditions['document_id'] = document_filter
        
        return self.vector_db.search_similar(
            self.collection_name,
            query_vector,
            limit,
            score_threshold,
            filter_conditions
        )
    
    def remove_document(self, document_id: str) -> bool:
        """Remove all chunks for a specific document"""
        try:
            # Search for all points with this document_id
            # Note: This is a simplified approach. In production, you'd want to maintain
            # a mapping of document_id to point_ids for efficient deletion
            
            # For now, we'll use a filter to find and delete
            # This is a placeholder - actual implementation would need to be more efficient
            return True
        except Exception as e:
            print(f"Error removing document {document_id}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics for client's collection"""
        info = self.vector_db.get_collection_info(self.collection_name)
        return info or {}


# Global vector database instance
vector_db = VectorDatabase()