import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO
from io import BytesIO
import aiofiles
from sqlalchemy.orm import Session

from ..models.client import Document, KnowledgeBase, DocumentChunk
from ..schemas.client import KnowledgeBaseConfig
from ..rag.chunking import ChunkingService, ChunkingConfig, ChunkingStrategy
from .vector_service import VectorService


class DocumentProcessor:
    """Handles document content extraction and preprocessing"""
    
    SUPPORTED_TYPES = {
        'text/plain': '.txt',
        'text/html': '.html',
        'text/markdown': '.md',
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/msword': '.doc',
        'application/json': '.json',
        'text/csv': '.csv'
    }
    
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB
    
    async def extract_content(self, file_data: bytes, content_type: str, filename: str) -> Dict[str, str]:
        """Extract text content from various file formats"""
        
        if len(file_data) > self.max_file_size:
            raise ValueError(f"File size exceeds limit of {self.max_file_size / 1024 / 1024}MB")
        
        if content_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {content_type}")
        
        try:
            if content_type == 'text/plain':
                return await self._extract_text(file_data)
            elif content_type in ['text/html', 'text/markdown']:
                return await self._extract_markup(file_data, content_type)
            elif content_type == 'application/pdf':
                return await self._extract_pdf(file_data)
            elif content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
                return await self._extract_docx(file_data)
            elif content_type == 'application/json':
                return await self._extract_json(file_data)
            elif content_type == 'text/csv':
                return await self._extract_csv(file_data)
            else:
                raise ValueError(f"Handler not implemented for: {content_type}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to extract content from {filename}: {str(e)}")
    
    async def _extract_text(self, file_data: bytes) -> Dict[str, str]:
        """Extract content from plain text files"""
        try:
            content = file_data.decode('utf-8')
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content = file_data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode text file")
        
        return {
            'content': content,
            'metadata': {
                'encoding': 'utf-8',
                'length': len(content)
            }
        }
    
    async def _extract_markup(self, file_data: bytes, content_type: str) -> Dict[str, str]:
        """Extract content from HTML/Markdown files"""
        content = file_data.decode('utf-8')
        
        if content_type == 'text/html':
            # Basic HTML tag removal (in production, use BeautifulSoup)
            import re
            text_content = re.sub(r'<[^>]+>', '', content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
        else:
            text_content = content
        
        return {
            'content': text_content,
            'metadata': {
                'original_format': content_type,
                'length': len(text_content)
            }
        }
    
    async def _extract_pdf(self, file_data: bytes) -> Dict[str, str]:
        """Extract content from PDF files"""
        # Note: In production, use PyPDF2 or pdfplumber
        # For now, return placeholder
        return {
            'content': "PDF extraction not yet implemented. Install PyPDF2 or pdfplumber.",
            'metadata': {
                'format': 'pdf',
                'extraction_method': 'placeholder',
                'note': 'Requires PDF processing library'
            }
        }
    
    async def _extract_docx(self, file_data: bytes) -> Dict[str, str]:
        """Extract content from DOCX files"""
        # Note: In production, use python-docx
        # For now, return placeholder
        return {
            'content': "DOCX extraction not yet implemented. Install python-docx.",
            'metadata': {
                'format': 'docx',
                'extraction_method': 'placeholder',
                'note': 'Requires DOCX processing library'
            }
        }
    
    async def _extract_json(self, file_data: bytes) -> Dict[str, str]:
        """Extract content from JSON files"""
        import json
        
        data = json.loads(file_data.decode('utf-8'))
        
        # Convert JSON to readable text
        if isinstance(data, dict):
            content_parts = []
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    content_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
                else:
                    content_parts.append(f"{key}: {json.dumps(value)}")
            content = '\n'.join(content_parts)
        else:
            content = json.dumps(data, indent=2)
        
        return {
            'content': content,
            'metadata': {
                'format': 'json',
                'original_structure': type(data).__name__,
                'length': len(content)
            }
        }
    
    async def _extract_csv(self, file_data: bytes) -> Dict[str, str]:
        """Extract content from CSV files"""
        import csv
        from io import StringIO
        
        content_str = file_data.decode('utf-8')
        csv_reader = csv.reader(StringIO(content_str))
        
        rows = list(csv_reader)
        if not rows:
            return {'content': '', 'metadata': {'format': 'csv', 'rows': 0}}
        
        # Convert CSV to readable text
        header = rows[0] if rows else []
        content_parts = [f"Headers: {', '.join(header)}"]
        
        for i, row in enumerate(rows[1:], 1):
            if i <= 100:  # Limit to first 100 rows for large files
                row_text = ', '.join(f"{header[j] if j < len(header) else f'col_{j}'}: {cell}" 
                                   for j, cell in enumerate(row))
                content_parts.append(f"Row {i}: {row_text}")
            else:
                content_parts.append(f"... and {len(rows) - 101} more rows")
                break
        
        content = '\n'.join(content_parts)
        
        return {
            'content': content,
            'metadata': {
                'format': 'csv',
                'rows': len(rows),
                'columns': len(header),
                'length': len(content)
            }
        }


class DocumentIngestionService:
    """Manages document ingestion workflow"""
    
    def __init__(self, db: Session, auto_generate_embeddings: bool = True):
        self.db = db
        self.processor = DocumentProcessor()
        self.chunking_service = ChunkingService()
        self.auto_generate_embeddings = auto_generate_embeddings
        self.vector_service = VectorService(db) if auto_generate_embeddings else None
    
    async def ingest_document(
        self,
        knowledge_base_id: int,
        file_data: bytes,
        filename: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
        source_url: Optional[str] = None
    ) -> Document:
        """Ingest a single document into a knowledge base"""
        
        # Get knowledge base
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if not kb:
            raise ValueError(f"Knowledge base {knowledge_base_id} not found")
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(filename)
        if not content_type:
            # Try to determine from content
            if filename.endswith('.md'):
                content_type = 'text/markdown'
            elif filename.endswith('.txt'):
                content_type = 'text/plain'
            else:
                content_type = 'application/octet-stream'
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(file_data).hexdigest()
        
        # Check for existing document with same hash
        existing_doc = self.db.query(Document).filter(
            Document.knowledge_base_id == knowledge_base_id,
            Document.content_hash == content_hash
        ).first()
        
        if existing_doc:
            return existing_doc
        
        # Extract content
        try:
            extracted = await self.processor.extract_content(file_data, content_type, filename)
        except Exception as e:
            # Create document with error status
            doc = Document(
                knowledge_base_id=knowledge_base_id,
                title=title or filename,
                content="",
                content_type=content_type,
                source_url=source_url,
                metadata=metadata or {},
                processing_status="failed",
                error_message=str(e),
                content_hash=content_hash
            )
            self.db.add(doc)
            self.db.commit()
            raise
        
        # Create document record
        doc = Document(
            knowledge_base_id=knowledge_base_id,
            title=title or filename,
            content=extracted['content'],
            content_type=content_type,
            source_url=source_url,
            metadata={**(metadata or {}), **extracted['metadata']},
            processing_status="completed",
            content_hash=content_hash
        )
        
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        
        # Chunk the document content
        await self._chunk_document(doc, kb)
        
        # Generate embeddings if enabled
        if self.auto_generate_embeddings and self.vector_service:
            try:
                # Ensure client collection exists
                await self.vector_service.create_client_collection(kb.client_id)
                
                # Generate embeddings for the document
                await self.vector_service.generate_and_store_embeddings(knowledge_base_id)
            except Exception as e:
                print(f"Warning: Failed to generate embeddings for document {doc.id}: {str(e)}")
                # Don't fail the ingestion if embeddings fail
        
        # Update knowledge base stats
        kb.total_documents = self.db.query(Document).filter(
            Document.knowledge_base_id == knowledge_base_id,
            Document.processing_status == "completed"
        ).count()
        
        kb.total_chunks = self.db.query(DocumentChunk).join(Document).filter(
            Document.knowledge_base_id == knowledge_base_id
        ).count()
        
        self.db.commit()
        
        return doc
    
    async def ingest_multiple_documents(
        self,
        knowledge_base_id: int,
        files: List[Dict[str, Union[bytes, str]]]
    ) -> List[Document]:
        """Ingest multiple documents in batch"""
        
        documents = []
        errors = []
        
        for file_info in files:
            try:
                doc = await self.ingest_document(
                    knowledge_base_id=knowledge_base_id,
                    file_data=file_info['data'],
                    filename=file_info['filename'],
                    title=file_info.get('title'),
                    metadata=file_info.get('metadata'),
                    source_url=file_info.get('source_url')
                )
                documents.append(doc)
            except Exception as e:
                errors.append({
                    'filename': file_info['filename'],
                    'error': str(e)
                })
        
        if errors:
            # Log errors but don't fail the entire batch
            print(f"Ingestion errors: {errors}")
        
        return documents
    
    async def update_document(
        self,
        document_id: int,
        file_data: Optional[bytes] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
        source_url: Optional[str] = None
    ) -> Document:
        """Update an existing document"""
        
        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document {document_id} not found")
        
        if file_data:
            # Re-extract content
            content_type = doc.content_type
            filename = doc.title
            
            content_hash = hashlib.sha256(file_data).hexdigest()
            
            try:
                extracted = await self.processor.extract_content(file_data, content_type, filename)
                doc.content = extracted['content']
                doc.metadata.update(extracted['metadata'])
                doc.content_hash = content_hash
                doc.processing_status = "completed"
                doc.error_message = None
                doc.version = str(float(doc.version) + 0.1)  # Increment version
            except Exception as e:
                doc.processing_status = "failed"
                doc.error_message = str(e)
                raise
        
        if title:
            doc.title = title
        
        if metadata:
            doc.metadata.update(metadata)
        
        if source_url:
            doc.source_url = source_url
        
        self.db.commit()
        self.db.refresh(doc)
        
        return doc
    
    async def delete_document(self, document_id: int) -> bool:
        """Delete a document and its chunks"""
        
        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False
        
        knowledge_base_id = doc.knowledge_base_id
        
        # Delete embeddings if vector service is available
        if self.vector_service:
            try:
                await self.vector_service.delete_document_embeddings(document_id)
            except Exception as e:
                print(f"Warning: Failed to delete embeddings for document {document_id}: {str(e)}")
        
        # Delete document (cascades to chunks via database foreign key)
        self.db.delete(doc)
        
        # Update knowledge base stats
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if kb:
            kb.total_documents = self.db.query(Document).filter(
                Document.knowledge_base_id == knowledge_base_id,
                Document.processing_status == "completed"
            ).count()
        
        self.db.commit()
        
        return True
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types"""
        return list(self.processor.SUPPORTED_TYPES.keys())
    
    def get_document_status(self, document_id: int) -> Optional[Dict]:
        """Get processing status of a document"""
        
        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return None
        
        return {
            'id': doc.id,
            'title': doc.title,
            'status': doc.processing_status,
            'error_message': doc.error_message,
            'content_length': len(doc.content),
            'chunks_count': len(doc.chunks),
            'metadata': doc.metadata,
            'created_at': doc.created_at,
            'updated_at': doc.updated_at
        }
    
    async def _chunk_document(self, document: Document, knowledge_base: KnowledgeBase):
        """Chunk document content and store chunks"""
        
        # Get chunking configuration from knowledge base
        chunking_config_dict = knowledge_base.chunking_config or {}
        
        # Create chunking configuration
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy(chunking_config_dict.get('chunking_strategy', 'semantic')),
            max_chunk_size=chunking_config_dict.get('chunk_size', 1000),
            min_chunk_size=chunking_config_dict.get('min_chunk_size', 100),
            overlap_size=chunking_config_dict.get('chunk_overlap', 200),
            respect_sentence_boundaries=chunking_config_dict.get('respect_sentence_boundaries', True),
            respect_paragraph_boundaries=chunking_config_dict.get('respect_paragraph_boundaries', True),
            preserve_headers=chunking_config_dict.get('preserve_headers', True)
        )
        
        # Prepare document metadata for chunking
        document_metadata = {
            'document_id': document.id,
            'document_title': document.title,
            'document_type': document.content_type,
            'source_url': document.source_url,
            **document.metadata
        }
        
        # Chunk the document
        chunks = self.chunking_service.chunk_document(
            text=document.content,
            config=chunking_config,
            document_metadata=document_metadata
        )
        
        # Delete existing chunks for this document (in case of re-processing)
        self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).delete()
        
        # Store chunks in database
        for chunk in chunks:
            doc_chunk = DocumentChunk(
                document_id=document.id,
                chunk_text=chunk.text,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
                vector_id=None  # Will be set when embeddings are generated
            )
            self.db.add(doc_chunk)
        
        self.db.commit()
        
        return chunks
    
    async def rechunk_document(self, document_id: int, chunking_config: ChunkingConfig = None) -> List[DocumentChunk]:
        """Re-chunk an existing document with new configuration"""
        
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        knowledge_base = self.db.query(KnowledgeBase).filter(
            KnowledgeBase.id == document.knowledge_base_id
        ).first()
        
        if chunking_config:
            # Update knowledge base chunking config
            knowledge_base.chunking_config.update(chunking_config.__dict__)
            self.db.commit()
        
        # Re-chunk the document
        chunks = await self._chunk_document(document, knowledge_base)
        
        # Return the database chunks
        return self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()
    
    def get_document_chunks(self, document_id: int) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        return self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()
    
    def get_chunk_statistics(self, knowledge_base_id: int) -> Dict[str, Any]:
        """Get chunking statistics for a knowledge base"""
        
        chunks = self.db.query(DocumentChunk).join(Document).filter(
            Document.knowledge_base_id == knowledge_base_id
        ).all()
        
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'chunk_types': []
            }
        
        # Convert to chunking service format for statistics
        from ..rag.chunking import TextChunk
        text_chunks = [
            TextChunk(
                text=chunk.chunk_text,
                start_char=0,  # Not tracked in DB
                end_char=len(chunk.chunk_text),
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]
        
        return self.chunking_service.get_chunk_statistics(text_chunks)