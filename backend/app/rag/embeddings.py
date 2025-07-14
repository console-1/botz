from typing import List, Dict, Any, Optional, Union
import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Sentence transformers imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# HuggingFace transformers imports
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    provider: str = "sentence_transformers"  # sentence_transformers, openai, huggingface
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 30
    normalize: bool = True
    cache_embeddings: bool = True


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: List[float]
    model_name: str
    provider: str
    metadata: Dict[str, Any]
    
    @property
    def dimension(self) -> int:
        return len(self.embedding)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'embedding': self.embedding,
            'model_name': self.model_name,
            'provider': self.provider,
            'dimension': self.dimension,
            'metadata': self.metadata
        }


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass


class SentenceTransformersProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for this provider")
        
        self.model = SentenceTransformer(config.model_name)
        
        # Enable GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using sentence transformers"""
        
        if not texts:
            return []
        
        try:
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    self._encode_batch, 
                    batch
                )
                
                all_embeddings.extend(embeddings)
            
            # Create results
            results = []
            for text, embedding in zip(texts, all_embeddings):
                if self.config.normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding.tolist(),
                    model_name=self.config.model_name,
                    provider="sentence_transformers",
                    metadata={
                        'batch_size': self.config.batch_size,
                        'normalized': self.config.normalize
                    }
                ))
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts"""
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'provider': 'sentence_transformers',
            'model_name': self.config.model_name,
            'dimension': self.get_dimension(),
            'max_seq_length': getattr(self.model, 'max_seq_length', 512),
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'cpu'
        }


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required for this provider")
        
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = config.api_key
        
        # Default to text-embedding-ada-002 if not specified
        if not config.model_name or config.model_name == "sentence-transformers/all-MiniLM-L6-v2":
            self.config.model_name = "text-embedding-ada-002"
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using OpenAI API"""
        
        if not texts:
            return []
        
        try:
            results = []
            
            # Process in batches (OpenAI has limits)
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                # Make API call
                for retry in range(self.config.max_retries):
                    try:
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: openai.Embedding.create(
                                input=batch,
                                model=self.config.model_name
                            )
                        )
                        break
                    except Exception as e:
                        if retry == self.config.max_retries - 1:
                            raise e
                        await asyncio.sleep(2 ** retry)  # Exponential backoff
                
                # Process response
                for j, embedding_data in enumerate(response['data']):
                    embedding = embedding_data['embedding']
                    
                    if self.config.normalize:
                        embedding = np.array(embedding)
                        embedding = embedding / np.linalg.norm(embedding)
                        embedding = embedding.tolist()
                    
                    results.append(EmbeddingResult(
                        text=batch[j],
                        embedding=embedding,
                        model_name=self.config.model_name,
                        provider="openai",
                        metadata={
                            'total_tokens': response.get('usage', {}).get('total_tokens', 0),
                            'model': response.get('model', self.config.model_name)
                        }
                    ))
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate OpenAI embeddings: {str(e)}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension for OpenAI models"""
        model_dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        return model_dimensions.get(self.config.model_name, 1536)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'provider': 'openai',
            'model_name': self.config.model_name,
            'dimension': self.get_dimension(),
            'max_tokens': 8191,  # OpenAI limit
            'pricing_per_1k_tokens': 0.0001  # Approximate
        }


class HuggingFaceProvider(EmbeddingProvider):
    """HuggingFace transformers embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for this provider")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        
        # Enable GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using HuggingFace transformers"""
        
        if not texts:
            return []
        
        try:
            results = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                # Run in thread pool
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    self._encode_batch,
                    batch
                )
                
                # Create results
                for text, embedding in zip(batch, embeddings):
                    if self.config.normalize:
                        embedding = embedding / np.linalg.norm(embedding)
                    
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=embedding.tolist(),
                        model_name=self.config.model_name,
                        provider="huggingface",
                        metadata={
                            'batch_size': self.config.batch_size,
                            'normalized': self.config.normalize
                        }
                    ))
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate HuggingFace embeddings: {str(e)}")
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts"""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Move back to CPU
            if torch.cuda.is_available():
                embeddings = embeddings.cpu()
            
            return embeddings.numpy()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.config.hidden_size
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'provider': 'huggingface',
            'model_name': self.config.model_name,
            'dimension': self.get_dimension(),
            'max_length': 512,
            'vocab_size': self.model.config.vocab_size
        }


class EmbeddingService:
    """Service for managing embedding generation"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._provider = None
        self._cache = {} if config and config.cache_embeddings else None
    
    def _get_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider"""
        if self._provider is None:
            if self.config.provider == "sentence_transformers":
                self._provider = SentenceTransformersProvider(self.config)
            elif self.config.provider == "openai":
                self._provider = OpenAIProvider(self.config)
            elif self.config.provider == "huggingface":
                self._provider = HuggingFaceProvider(self.config)
            else:
                raise ValueError(f"Unknown embedding provider: {self.config.provider}")
        
        return self._provider
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for a list of texts"""
        
        if not texts:
            return []
        
        # Check cache if enabled
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        if self._cache is not None:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached_results.append((i, self._cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        new_results = []
        if uncached_texts:
            provider = self._get_provider()
            new_results = await provider.generate_embeddings(uncached_texts)
            
            # Cache new results
            if self._cache is not None:
                for text, result in zip(uncached_texts, new_results):
                    cache_key = self._get_cache_key(text)
                    self._cache[cache_key] = result
        
        # Combine cached and new results in original order
        all_results = [None] * len(texts)
        
        # Place cached results
        for original_index, cached_result in cached_results:
            all_results[original_index] = cached_result
        
        # Place new results
        for uncached_index, new_result in zip(uncached_indices, new_results):
            all_results[uncached_index] = new_result
        
        return all_results
    
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        results = await self.generate_embeddings([text])
        return results[0] if results else None
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use model name and text hash as key
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"{self.config.model_name}:{text_hash}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        provider = self._get_provider()
        return provider.get_model_info()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        provider = self._get_provider()
        return provider.get_dimension()
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self._cache is not None:
            self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings"""
        return len(self._cache) if self._cache is not None else 0
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available embedding providers"""
        providers = []
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            providers.append("sentence_transformers")
        
        if OPENAI_AVAILABLE:
            providers.append("openai")
        
        if TRANSFORMERS_AVAILABLE:
            providers.append("huggingface")
        
        return providers
    
    @staticmethod
    def get_recommended_models() -> Dict[str, List[str]]:
        """Get recommended models for each provider"""
        return {
            "sentence_transformers": [
                "sentence-transformers/all-MiniLM-L6-v2",  # Fast, good quality
                "sentence-transformers/all-mpnet-base-v2",  # Better quality
                "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # QA optimized
            ],
            "openai": [
                "text-embedding-ada-002",  # Most popular
                "text-embedding-3-small",  # Newer, efficient
                "text-embedding-3-large"   # Best quality
            ],
            "huggingface": [
                "microsoft/DialoGPT-medium",
                "distilbert-base-uncased",
                "roberta-base"
            ]
        }


# Utility functions

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    a = np.array(embedding1)
    b = np.array(embedding2)
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(query_embedding: List[float], embeddings: List[List[float]]) -> List[float]:
    """Calculate cosine similarity between query and multiple embeddings"""
    query = np.array(query_embedding)
    embeddings_matrix = np.array(embeddings)
    
    # Normalize
    query_norm = query / np.linalg.norm(query)
    embeddings_norm = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    
    # Calculate similarities
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities.tolist()


def find_most_similar(
    query_embedding: List[float], 
    embeddings: List[List[float]], 
    top_k: int = 5
) -> List[tuple]:
    """Find most similar embeddings to query"""
    similarities = batch_cosine_similarity(query_embedding, embeddings)
    
    # Get top-k indices and scores
    indexed_similarities = [(i, score) for i, score in enumerate(similarities)]
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)
    
    return indexed_similarities[:top_k]