import pytest
import asyncio
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.client import Client, KnowledgeBase, Document
from app.services.document_ingestion import DocumentProcessor, DocumentIngestionService


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a test database session"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_client(db_session):
    """Create a test client"""
    client = Client(
        client_id="test_client",
        name="Test Client",
        email="test@example.com",
        config={},
        branding={},
        features={}
    )
    db_session.add(client)
    db_session.commit()
    db_session.refresh(client)
    return client


@pytest.fixture
def test_knowledge_base(db_session, test_client):
    """Create a test knowledge base"""
    kb = KnowledgeBase(
        client_id=test_client.client_id,
        name="Test KB",
        description="Test knowledge base",
        chunking_config={},
        embedding_config={}
    )
    db_session.add(kb)
    db_session.commit()
    db_session.refresh(kb)
    return kb


class TestDocumentProcessor:
    """Test document processing functionality"""
    
    def setup_method(self):
        self.processor = DocumentProcessor()
    
    @pytest.mark.asyncio
    async def test_extract_text_content(self):
        """Test text file content extraction"""
        content = "This is a test document with some content."
        file_data = content.encode('utf-8')
        
        result = await self.processor.extract_content(file_data, 'text/plain', 'test.txt')
        
        assert result['content'] == content
        assert result['metadata']['encoding'] == 'utf-8'
        assert result['metadata']['length'] == len(content)
    
    @pytest.mark.asyncio
    async def test_extract_html_content(self):
        """Test HTML content extraction"""
        html_content = "<html><body><h1>Title</h1><p>Paragraph content</p></body></html>"
        file_data = html_content.encode('utf-8')
        
        result = await self.processor.extract_content(file_data, 'text/html', 'test.html')
        
        # Should strip HTML tags
        assert '<' not in result['content']
        assert 'Title' in result['content']
        assert 'Paragraph content' in result['content']
    
    @pytest.mark.asyncio
    async def test_extract_json_content(self):
        """Test JSON content extraction"""
        json_content = '{"title": "Test", "description": "A test document", "items": ["item1", "item2"]}'
        file_data = json_content.encode('utf-8')
        
        result = await self.processor.extract_content(file_data, 'application/json', 'test.json')
        
        assert 'title: Test' in result['content']
        assert 'description: A test document' in result['content']
        assert 'items: item1, item2' in result['content']
    
    @pytest.mark.asyncio
    async def test_extract_csv_content(self):
        """Test CSV content extraction"""
        csv_content = "name,age,city\nJohn,25,New York\nJane,30,Los Angeles"
        file_data = csv_content.encode('utf-8')
        
        result = await self.processor.extract_content(file_data, 'text/csv', 'test.csv')
        
        assert 'Headers: name, age, city' in result['content']
        assert 'Row 1: name: John, age: 25, city: New York' in result['content']
        assert result['metadata']['rows'] == 3  # Including header
        assert result['metadata']['columns'] == 3
    
    @pytest.mark.asyncio
    async def test_unsupported_file_type(self):
        """Test handling of unsupported file types"""
        file_data = b"some binary data"
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            await self.processor.extract_content(file_data, 'application/octet-stream', 'test.bin')
    
    @pytest.mark.asyncio
    async def test_file_size_limit(self):
        """Test file size limit enforcement"""
        # Create large file data
        large_data = b"x" * (51 * 1024 * 1024)  # 51MB
        
        with pytest.raises(ValueError, match="File size exceeds limit"):
            await self.processor.extract_content(large_data, 'text/plain', 'large.txt')


class TestDocumentIngestionService:
    """Test document ingestion service functionality"""
    
    @pytest.mark.asyncio
    async def test_ingest_document_success(self, db_session, test_knowledge_base):
        """Test successful document ingestion"""
        service = DocumentIngestionService(db_session)
        
        content = "This is a test document for ingestion."
        file_data = content.encode('utf-8')
        
        document = await service.ingest_document(
            knowledge_base_id=test_knowledge_base.id,
            file_data=file_data,
            filename="test.txt",
            title="Test Document"
        )
        
        assert document.id is not None
        assert document.title == "Test Document"
        assert document.content == content
        assert document.content_type == "text/plain"
        assert document.processing_status == "completed"
        assert document.content_hash is not None
    
    @pytest.mark.asyncio
    async def test_ingest_duplicate_document(self, db_session, test_knowledge_base):
        """Test handling of duplicate documents"""
        service = DocumentIngestionService(db_session)
        
        content = "This is a duplicate test document."
        file_data = content.encode('utf-8')
        
        # Ingest first time
        doc1 = await service.ingest_document(
            knowledge_base_id=test_knowledge_base.id,
            file_data=file_data,
            filename="test.txt"
        )
        
        # Ingest same content again
        doc2 = await service.ingest_document(
            knowledge_base_id=test_knowledge_base.id,
            file_data=file_data,
            filename="test_copy.txt"
        )
        
        # Should return the same document
        assert doc1.id == doc2.id
        assert doc1.content_hash == doc2.content_hash
    
    @pytest.mark.asyncio
    async def test_ingest_document_invalid_kb(self, db_session):
        """Test ingestion with invalid knowledge base"""
        service = DocumentIngestionService(db_session)
        
        file_data = b"test content"
        
        with pytest.raises(ValueError, match="Knowledge base .* not found"):
            await service.ingest_document(
                knowledge_base_id=999,  # Non-existent KB
                file_data=file_data,
                filename="test.txt"
            )
    
    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(self, db_session, test_knowledge_base):
        """Test batch document ingestion"""
        service = DocumentIngestionService(db_session)
        
        files = [
            {
                'data': b"First document content",
                'filename': "doc1.txt",
                'title': "Document 1"
            },
            {
                'data': b"Second document content",
                'filename': "doc2.txt",
                'title': "Document 2"
            }
        ]
        
        documents = await service.ingest_multiple_documents(
            knowledge_base_id=test_knowledge_base.id,
            files=files
        )
        
        assert len(documents) == 2
        assert documents[0].title == "Document 1"
        assert documents[1].title == "Document 2"
        assert all(doc.processing_status == "completed" for doc in documents)
    
    @pytest.mark.asyncio
    async def test_update_document(self, db_session, test_knowledge_base):
        """Test document update functionality"""
        service = DocumentIngestionService(db_session)
        
        # Create initial document
        original_content = "Original content"
        doc = await service.ingest_document(
            knowledge_base_id=test_knowledge_base.id,
            file_data=original_content.encode('utf-8'),
            filename="test.txt",
            title="Original Title"
        )
        
        original_version = doc.version
        
        # Update document
        new_content = "Updated content"
        updated_doc = await service.update_document(
            document_id=doc.id,
            file_data=new_content.encode('utf-8'),
            title="Updated Title"
        )
        
        assert updated_doc.id == doc.id
        assert updated_doc.content == new_content
        assert updated_doc.title == "Updated Title"
        assert float(updated_doc.version) > float(original_version)
    
    @pytest.mark.asyncio
    async def test_delete_document(self, db_session, test_knowledge_base):
        """Test document deletion"""
        service = DocumentIngestionService(db_session)
        
        # Create document
        doc = await service.ingest_document(
            knowledge_base_id=test_knowledge_base.id,
            file_data=b"test content",
            filename="test.txt"
        )
        
        # Delete document
        success = await service.delete_document(doc.id)
        
        assert success is True
        
        # Verify document is deleted
        deleted_doc = db_session.query(Document).filter(Document.id == doc.id).first()
        assert deleted_doc is None
    
    def test_get_document_status(self, db_session, test_knowledge_base):
        """Test document status retrieval"""
        service = DocumentIngestionService(db_session)
        
        # Create document directly in database
        doc = Document(
            knowledge_base_id=test_knowledge_base.id,
            title="Test Doc",
            content="Test content",
            content_type="text/plain",
            processing_status="completed",
            content_hash="abc123"
        )
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)
        
        status = service.get_document_status(doc.id)
        
        assert status is not None
        assert status['id'] == doc.id
        assert status['title'] == "Test Doc"
        assert status['status'] == "completed"
        assert status['content_length'] == len("Test content")
    
    def test_get_supported_file_types(self):
        """Test supported file types listing"""
        service = DocumentIngestionService(MagicMock())
        
        supported_types = service.get_supported_file_types()
        
        assert 'text/plain' in supported_types
        assert 'text/html' in supported_types
        assert 'application/json' in supported_types
        assert 'text/csv' in supported_types


if __name__ == "__main__":
    pytest.main([__file__])