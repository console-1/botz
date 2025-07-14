import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Download required NLTK data (should be done during setup)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SECTION = "section"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 200
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    preserve_headers: bool = True
    language: str = "english"


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: Dict[str, Any]
    
    @property
    def size(self) -> int:
        return len(self.text)


class SemanticChunker:
    """Intelligent text chunking that preserves semantic boundaries"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        
        # Patterns for detecting semantic boundaries
        self.header_patterns = [
            r'^#{1,6}\s+.+$',  # Markdown headers
            r'^[A-Z][A-Z\s]+:?\s*$',  # ALL CAPS headers
            r'^\d+\.\s+.+$',  # Numbered sections
            r'^[A-Z][^.!?]*:$',  # Title case with colon
        ]
        
        self.section_break_patterns = [
            r'\n\s*\n\s*\n',  # Multiple line breaks
            r'^\s*[-=]{3,}\s*$',  # Horizontal rules
            r'^\s*\*{3,}\s*$',  # Asterisk separators
        ]
        
        self.sentence_ending_patterns = [
            r'[.!?]+\s+',
            r'[.!?]+$'
        ]
    
    def chunk_text(self, text: str, document_metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Chunk text using the configured strategy
        
        Args:
            text: Input text to chunk
            document_metadata: Additional metadata about the document
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        document_metadata = document_metadata or {}
        
        if self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text, document_metadata)
        elif self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(text, document_metadata)
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(text, document_metadata)
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(text, document_metadata)
        elif self.config.strategy == ChunkingStrategy.SECTION:
            return self._section_chunking(text, document_metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
    
    def _semantic_chunking(self, text: str, document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Intelligent semantic chunking that preserves meaning boundaries
        """
        chunks = []
        
        # First, identify major sections
        sections = self._identify_sections(text)
        
        chunk_index = 0
        for section in sections:
            section_chunks = self._chunk_section_semantically(
                section, chunk_index, document_metadata
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify major sections in the text"""
        sections = []
        lines = text.split('\n')
        current_section = {'start': 0, 'lines': [], 'header': None}
        
        for i, line in enumerate(lines):
            is_header = self._is_header(line)
            is_section_break = self._is_section_break(line)
            
            if is_header and current_section['lines']:
                # Save current section and start new one
                sections.append(current_section)
                current_section = {
                    'start': len('\n'.join(current_section['lines'])) + 1,
                    'lines': [line],
                    'header': line.strip()
                }
            elif is_section_break and current_section['lines']:
                # Save current section and start new one
                sections.append(current_section)
                current_section = {
                    'start': len('\n'.join(current_section['lines'])) + 1,
                    'lines': [],
                    'header': None
                }
            else:
                current_section['lines'].append(line)
        
        # Add the last section
        if current_section['lines']:
            sections.append(current_section)
        
        # Convert to text sections
        text_sections = []
        char_offset = 0
        
        for section in sections:
            section_text = '\n'.join(section['lines'])
            if section_text.strip():
                text_sections.append({
                    'text': section_text,
                    'start_char': char_offset,
                    'end_char': char_offset + len(section_text),
                    'header': section['header'],
                    'metadata': {'section_type': 'header' if section['header'] else 'content'}
                })
            char_offset += len(section_text) + 1  # +1 for newline
        
        return text_sections
    
    def _chunk_section_semantically(
        self, 
        section: Dict[str, Any], 
        starting_chunk_index: int,
        document_metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Chunk a section using semantic boundaries"""
        text = section['text']
        chunks = []
        
        # If section is small enough, return as single chunk
        if len(text) <= self.config.max_chunk_size:
            chunk_metadata = {
                **document_metadata,
                'section_header': section.get('header'),
                'chunk_type': 'semantic_section',
                **section.get('metadata', {})
            }
            
            chunks.append(TextChunk(
                text=text.strip(),
                start_char=section['start_char'],
                end_char=section['end_char'],
                chunk_index=starting_chunk_index,
                metadata=chunk_metadata
            ))
            return chunks
        
        # For larger sections, split by paragraphs and sentences
        paragraphs = self._split_into_paragraphs(text)
        current_chunk = ""
        current_start = section['start_char']
        chunk_index = starting_chunk_index
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, finalize current chunk
            if (current_chunk and 
                len(current_chunk) + len(paragraph) + 1 > self.config.max_chunk_size):
                
                # Create chunk with overlap if configured
                chunk_text = current_chunk.strip()
                if chunk_text:
                    chunk_metadata = {
                        **document_metadata,
                        'section_header': section.get('header'),
                        'chunk_type': 'semantic_paragraph',
                        **section.get('metadata', {})
                    }
                    
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        chunk_index=chunk_index,
                        metadata=chunk_metadata
                    ))
                    
                    # Handle overlap
                    if self.config.overlap_size > 0:
                        overlap_text = self._get_overlap_text(chunk_text, self.config.overlap_size)
                        current_chunk = overlap_text + " " + paragraph
                    else:
                        current_chunk = paragraph
                    
                    current_start += len(chunk_text) - len(overlap_text) if self.config.overlap_size > 0 else len(chunk_text)
                    chunk_index += 1
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the final chunk
        if current_chunk.strip():
            chunk_metadata = {
                **document_metadata,
                'section_header': section.get('header'),
                'chunk_type': 'semantic_final',
                **section.get('metadata', {})
            }
            
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                chunk_index=chunk_index,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _fixed_size_chunking(self, text: str, document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """Simple fixed-size chunking with overlap"""
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            end = min(start + self.config.max_chunk_size, len(text))
            
            # Respect sentence boundaries if configured
            if (self.config.respect_sentence_boundaries and 
                end < len(text) and 
                not self._is_sentence_boundary(text, end)):
                # Find the nearest sentence boundary
                sentence_end = self._find_sentence_boundary(text, start, end)
                if sentence_end > start + self.config.min_chunk_size:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **document_metadata,
                    'chunk_type': 'fixed_size',
                    'chunk_method': 'character_count'
                }
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=chunk_metadata
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(end - self.config.overlap_size, start + 1)
        
        return chunks
    
    def _paragraph_chunking(self, text: str, document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunk by paragraphs, combining small ones"""
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            if (current_chunk and 
                len(current_chunk) + len(paragraph) > self.config.max_chunk_size):
                
                # Finalize current chunk
                chunk_metadata = {
                    **document_metadata,
                    'chunk_type': 'paragraph',
                    'paragraph_count': current_chunk.count('\n\n') + 1
                }
                
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    chunk_index=chunk_index,
                    metadata=chunk_metadata
                ))
                
                current_chunk = paragraph
                current_start += len(current_chunk)
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = {
                **document_metadata,
                'chunk_type': 'paragraph',
                'paragraph_count': current_chunk.count('\n\n') + 1
            }
            
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                chunk_index=chunk_index,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _sentence_chunking(self, text: str, document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunk by sentences, combining to reach target size"""
        sentences = nltk.sent_tokenize(text, language=self.config.language)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            if (current_chunk and 
                len(current_chunk) + len(sentence) > self.config.max_chunk_size):
                
                # Finalize current chunk
                chunk_metadata = {
                    **document_metadata,
                    'chunk_type': 'sentence',
                    'sentence_count': len(nltk.sent_tokenize(current_chunk))
                }
                
                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    chunk_index=chunk_index,
                    metadata=chunk_metadata
                ))
                
                current_chunk = sentence
                current_start += len(current_chunk)
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = {
                **document_metadata,
                'chunk_type': 'sentence',
                'sentence_count': len(nltk.sent_tokenize(current_chunk))
            }
            
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                chunk_index=chunk_index,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _section_chunking(self, text: str, document_metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunk by identified sections (headers, breaks, etc.)"""
        sections = self._identify_sections(text)
        chunks = []
        
        for i, section in enumerate(sections):
            chunk_metadata = {
                **document_metadata,
                'chunk_type': 'section',
                'section_header': section.get('header'),
                'section_index': i
            }
            
            chunks.append(TextChunk(
                text=section['text'].strip(),
                start_char=section['start_char'],
                end_char=section['end_char'],
                chunk_index=i,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    # Helper methods
    
    def _is_header(self, line: str) -> bool:
        """Check if a line is a header"""
        line = line.strip()
        if not line:
            return False
        
        for pattern in self.header_patterns:
            if re.match(pattern, line, re.MULTILINE):
                return True
        
        return False
    
    def _is_section_break(self, line: str) -> bool:
        """Check if a line is a section break"""
        for pattern in self.section_break_patterns:
            if re.match(pattern, line):
                return True
        return False
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines, but preserve single newlines within paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _is_sentence_boundary(self, text: str, position: int) -> bool:
        """Check if position is at a sentence boundary"""
        if position >= len(text):
            return True
        
        # Look backwards for sentence ending
        for i in range(min(50, position), 0, -1):
            char = text[position - i]
            if char in '.!?':
                # Check if followed by whitespace or end of text
                next_pos = position - i + 1
                if next_pos >= len(text) or text[next_pos].isspace():
                    return True
        
        return False
    
    def _find_sentence_boundary(self, text: str, start: int, preferred_end: int) -> int:
        """Find the nearest sentence boundary before preferred_end"""
        # Look backwards from preferred_end
        for i in range(preferred_end - 1, start, -1):
            if self._is_sentence_boundary(text, i):
                return i
        
        return preferred_end
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk"""
        if overlap_size <= 0 or len(text) <= overlap_size:
            return text
        
        # Try to get overlap at sentence boundary
        overlap_start = len(text) - overlap_size
        sentences = nltk.sent_tokenize(text[overlap_start:])
        
        if len(sentences) > 1:
            # Return the last complete sentence(s) that fit in overlap_size
            return sentences[-1]
        else:
            # Return the last overlap_size characters
            return text[-overlap_size:]


class ChunkingService:
    """Service for managing text chunking operations"""
    
    def __init__(self, default_config: ChunkingConfig = None):
        self.default_config = default_config or ChunkingConfig()
        self.chunkers = {}
    
    def get_chunker(self, config: ChunkingConfig = None) -> SemanticChunker:
        """Get a chunker instance with the specified configuration"""
        config = config or self.default_config
        config_key = (config.strategy, config.max_chunk_size, config.overlap_size)
        
        if config_key not in self.chunkers:
            self.chunkers[config_key] = SemanticChunker(config)
        
        return self.chunkers[config_key]
    
    def chunk_document(
        self, 
        text: str, 
        config: ChunkingConfig = None,
        document_metadata: Dict[str, Any] = None
    ) -> List[TextChunk]:
        """Chunk a document with the specified configuration"""
        chunker = self.get_chunker(config)
        return chunker.chunk_text(text, document_metadata)
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about a set of chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        sizes = [chunk.size for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(sizes),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'chunk_types': list(set(chunk.metadata.get('chunk_type', 'unknown') for chunk in chunks))
        }