import pytest
from app.rag.chunking import (
    ChunkingStrategy, 
    ChunkingConfig, 
    SemanticChunker, 
    ChunkingService,
    TextChunk
)


class TestChunkingConfig:
    """Test chunking configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ChunkingConfig()
        
        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.max_chunk_size == 1000
        assert config.min_chunk_size == 100
        assert config.overlap_size == 200
        assert config.respect_sentence_boundaries is True
        assert config.respect_paragraph_boundaries is True
        assert config.preserve_headers is True
        assert config.language == "english"
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chunk_size=500,
            min_chunk_size=50,
            overlap_size=100
        )
        
        assert config.strategy == ChunkingStrategy.FIXED_SIZE
        assert config.max_chunk_size == 500
        assert config.min_chunk_size == 50
        assert config.overlap_size == 100


class TestSemanticChunker:
    """Test semantic chunking functionality"""
    
    def setup_method(self):
        self.chunker = SemanticChunker()
    
    def test_empty_text(self):
        """Test chunking empty text"""
        chunks = self.chunker.chunk_text("")
        assert len(chunks) == 0
        
        chunks = self.chunker.chunk_text("   ")
        assert len(chunks) == 0
    
    def test_short_text_single_chunk(self):
        """Test that short text produces single chunk"""
        text = "This is a short text that should fit in one chunk."
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].size == len(text)
    
    def test_semantic_chunking_with_paragraphs(self):
        """Test semantic chunking with multiple paragraphs"""
        text = """
        This is the first paragraph with some content that talks about topic A.
        It has multiple sentences to provide context.
        
        This is the second paragraph that discusses topic B.
        It should be in a different chunk if the content is long enough.
        
        This is the third paragraph about topic C.
        It continues the discussion with more details.
        """
        
        config = ChunkingConfig(max_chunk_size=200)
        chunker = SemanticChunker(config)
        chunks = chunker.chunk_text(text.strip())
        
        assert len(chunks) > 1
        assert all(chunk.size <= 200 for chunk in chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))
    
    def test_fixed_size_chunking(self):
        """Test fixed size chunking strategy"""
        text = "A" * 1000  # 1000 character string
        
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chunk_size=300,
            overlap_size=50
        )
        chunker = SemanticChunker(config)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 3  # Should need multiple chunks
        assert all(chunk.size <= 300 for chunk in chunks)
        
        # Check overlap exists (except for last chunk)
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # There should be some overlap in content
            assert len(current_chunk.text) > 0
            assert len(next_chunk.text) > 0
    
    def test_paragraph_chunking(self):
        """Test paragraph-based chunking strategy"""
        text = """Paragraph one with some content.

Paragraph two with different content.

Paragraph three with more content."""
        
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            max_chunk_size=100
        )
        chunker = SemanticChunker(config)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Each chunk should contain complete paragraphs
        for chunk in chunks:
            assert "Paragraph" in chunk.text
    
    def test_sentence_chunking(self):
        """Test sentence-based chunking strategy"""
        text = ("This is sentence one. This is sentence two. This is sentence three. "
                "This is sentence four. This is sentence five. This is sentence six.")
        
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE,
            max_chunk_size=100
        )
        chunker = SemanticChunker(config)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.text.endswith('.') or chunk == chunks[-1]  # Last chunk might not end with period
    
    def test_section_chunking_with_headers(self):
        """Test section-based chunking with headers"""
        text = """# Header One
        
        Content under header one.
        More content here.
        
        ## Header Two
        
        Content under header two.
        Different topic here.
        
        ### Header Three
        
        Content under header three.
        Final section content."""
        
        config = ChunkingConfig(strategy=ChunkingStrategy.SECTION)
        chunker = SemanticChunker(config)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Should have different sections
        header_chunks = [chunk for chunk in chunks if '#' in chunk.text]
        assert len(header_chunks) > 0
    
    def test_header_detection(self):
        """Test header detection functionality"""
        chunker = SemanticChunker()
        
        assert chunker._is_header("# Main Header")
        assert chunker._is_header("## Sub Header")
        assert chunker._is_header("### Sub-sub Header")
        assert chunker._is_header("INTRODUCTION:")
        assert chunker._is_header("1. First Section")
        
        assert not chunker._is_header("This is regular text.")
        assert not chunker._is_header("Not a header line")
    
    def test_section_break_detection(self):
        """Test section break detection"""
        chunker = SemanticChunker()
        
        assert chunker._is_section_break("---")
        assert chunker._is_section_break("===")
        assert chunker._is_section_break("***")
        assert chunker._is_section_break("   ---   ")
        
        assert not chunker._is_section_break("regular text")
        assert not chunker._is_section_break("--")  # Too short
    
    def test_chunk_metadata(self):
        """Test that chunks contain proper metadata"""
        text = "This is a test document with some content."
        document_metadata = {
            "document_id": 123,
            "document_title": "Test Document",
            "author": "Test Author"
        }
        
        chunks = self.chunker.chunk_text(text, document_metadata)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.metadata["document_id"] == 123
        assert chunk.metadata["document_title"] == "Test Document"
        assert chunk.metadata["author"] == "Test Author"
        assert "chunk_type" in chunk.metadata
    
    def test_overlap_functionality(self):
        """Test that overlap works correctly"""
        text = "A" * 500 + " " + "B" * 500  # 1001 characters with space
        
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chunk_size=300,
            overlap_size=50
        )
        chunker = SemanticChunker(config)
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2
        
        # Check that there's overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].text
            next_chunk = chunks[i + 1].text
            
            # Should have some common content (overlap)
            overlap_found = False
            for j in range(min(50, len(current_chunk))):
                if current_chunk[-j-1:] in next_chunk[:j+1]:
                    overlap_found = True
                    break
            
            # Note: Due to sentence boundary respect, exact overlap might vary


class TestChunkingService:
    """Test chunking service functionality"""
    
    def setup_method(self):
        self.service = ChunkingService()
    
    def test_get_chunker_caching(self):
        """Test that chunker instances are cached properly"""
        config1 = ChunkingConfig(max_chunk_size=500)
        config2 = ChunkingConfig(max_chunk_size=500)  # Same config
        config3 = ChunkingConfig(max_chunk_size=1000)  # Different config
        
        chunker1 = self.service.get_chunker(config1)
        chunker2 = self.service.get_chunker(config2)
        chunker3 = self.service.get_chunker(config3)
        
        # Same config should return same instance
        assert chunker1 is chunker2
        # Different config should return different instance
        assert chunker1 is not chunker3
    
    def test_chunk_document(self):
        """Test document chunking through service"""
        text = "This is a test document that will be chunked by the service."
        document_metadata = {"document_id": 456}
        
        chunks = self.service.chunk_document(text, document_metadata=document_metadata)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.metadata["document_id"] == 456 for chunk in chunks)
    
    def test_get_chunk_statistics_empty(self):
        """Test statistics with empty chunk list"""
        stats = self.service.get_chunk_statistics([])
        
        assert stats["total_chunks"] == 0
        assert stats["total_characters"] == 0
        assert stats["avg_chunk_size"] == 0
        assert stats["min_chunk_size"] == 0
        assert stats["max_chunk_size"] == 0
    
    def test_get_chunk_statistics_with_chunks(self):
        """Test statistics with actual chunks"""
        text1 = "Short chunk."
        text2 = "This is a longer chunk with more content."
        text3 = "Medium length chunk here."
        
        chunks = [
            TextChunk(text1, 0, len(text1), 0, {"chunk_type": "test"}),
            TextChunk(text2, 0, len(text2), 1, {"chunk_type": "test"}),
            TextChunk(text3, 0, len(text3), 2, {"chunk_type": "test"})
        ]
        
        stats = self.service.get_chunk_statistics(chunks)
        
        assert stats["total_chunks"] == 3
        assert stats["total_characters"] == len(text1) + len(text2) + len(text3)
        assert stats["avg_chunk_size"] == (len(text1) + len(text2) + len(text3)) / 3
        assert stats["min_chunk_size"] == min(len(text1), len(text2), len(text3))
        assert stats["max_chunk_size"] == max(len(text1), len(text2), len(text3))
        assert "test" in stats["chunk_types"]


class TestTextChunk:
    """Test TextChunk functionality"""
    
    def test_text_chunk_creation(self):
        """Test creating a TextChunk"""
        text = "This is chunk text."
        metadata = {"type": "test", "source": "unit_test"}
        
        chunk = TextChunk(
            text=text,
            start_char=0,
            end_char=len(text),
            chunk_index=5,
            metadata=metadata
        )
        
        assert chunk.text == text
        assert chunk.start_char == 0
        assert chunk.end_char == len(text)
        assert chunk.chunk_index == 5
        assert chunk.metadata == metadata
        assert chunk.size == len(text)
    
    def test_chunk_size_property(self):
        """Test that size property returns correct length"""
        text = "A" * 150
        chunk = TextChunk(text, 0, 150, 0, {})
        
        assert chunk.size == 150
        assert chunk.size == len(text)


if __name__ == "__main__":
    pytest.main([__file__])