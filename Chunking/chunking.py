#!/usr/bin/env python3
"""
Text Chunking System

A fast and efficient text chunking system for processing crawled web data:
- Sliding window chunking with overlap
- Text preprocessing and cleaning
- Metadata preservation
- Batch processing capabilities
- Configurable chunk parameters
- Export to various formats

Usage:
    from chunking import TextChunker
    
    chunker = TextChunker(chunk_size=1000, overlap=200)
    chunks = chunker.chunk_text("Your long text here...")
    
    # Process crawled data
    chunker.process_crawled_data(crawled_results)
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from datetime import datetime


@dataclass
class TextChunk:
    """Data class for text chunks."""
    id: str
    text: str
    chunk_index: int
    total_chunks: int
    source_url: str
    metadata: Dict[str, Any]
    chunk_size: int
    overlap_size: int
    created_at: str


@dataclass
class ChunkingConfig:
    """Configuration for chunking parameters."""
    chunk_size: int = 1000
    overlap_size: int = 200
    min_chunk_size: int = 100
    clean_text: bool = True
    normalize_whitespace: bool = True


class TextChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None, **kwargs):
        """
        Initialize the text chunker.
        
        Args:
            config: ChunkingConfig object with parameters
            **kwargs: Override config parameters
        """
        # Use provided config or create default
        if config:
            self.config = config
        else:
            self.config = ChunkingConfig()
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning."""
        self.patterns = {
            'multiple_spaces': re.compile(r'\s+')
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not self.config.clean_text:
            return text
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.patterns['multiple_spaces'].sub(' ', text)
            text = text.strip()
        
        return text
    

    

    

    
    def _chunk_text_with_sliding_window(self, text: str, source_url: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Sliding window chunking strategy with overlap.
        
        Args:
            text: Text to chunk
            source_url: Source URL
            metadata: Additional metadata
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        text_length = len(text)
        
        if text_length <= self.config.chunk_size:
            # Text is smaller than chunk size, create single chunk
            chunk_id = self._generate_chunk_id(source_url, 0)
            chunks.append(TextChunk(
                id=chunk_id,
                text=text,
                chunk_index=0,
                total_chunks=1,
                source_url=source_url,
                metadata=metadata.copy(),
                chunk_size=text_length,
                overlap_size=0,
                created_at=datetime.now().isoformat()
            ))
            return chunks
        
        # Use sliding window with overlap
        window_size = self.config.chunk_size
        step_size = window_size - self.config.overlap_size
        
        chunk_index = 0
        start = 0
        
        while start < text_length:
            end = min(start + window_size, text_length)
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk_id = self._generate_chunk_id(source_url, chunk_index)
                chunks.append(TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    source_url=source_url,
                    metadata=metadata.copy(),
                    chunk_size=len(chunk_text),
                    overlap_size=self.config.overlap_size,
                    created_at=datetime.now().isoformat()
                ))
                chunk_index += 1
            
            start += step_size
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def chunk_text(self, text: str, source_url: str = "", metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text using sliding window with overlap strategy.
        
        Args:
            text: Text to chunk
            source_url: Source URL (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Use sliding window chunking with overlap
        chunks = self._chunk_text_with_sliding_window(cleaned_text, source_url, metadata)
        
        return chunks
    
    def process_crawled_data(self, crawled_data: Dict[str, str]) -> List[TextChunk]:
        """
        Process crawled data from the web crawler.
        
        Args:
            crawled_data: Dictionary mapping URLs to text content
            
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        
        self.logger.info(f"Processing {len(crawled_data)} crawled pages")
        
        for url, text in crawled_data.items():
            try:
                # Create metadata for this page
                metadata = {
                    'url': url,
                    'text_length': len(text),
                    'processed_at': datetime.now().isoformat(),
                    'chunking_strategy': 'sliding_window_with_overlap'
                }
                
                # Chunk the text
                chunks = self.chunk_text(text, url, metadata)
                all_chunks.extend(chunks)
                
                self.logger.debug(f"Created {len(chunks)} chunks for {url}")
                
            except Exception as e:
                self.logger.error(f"Error processing {url}: {e}")
                continue
        
        self.logger.info(f"Created {len(all_chunks)} total chunks")
        
        return all_chunks
    

    

    
    def _generate_chunk_id(self, source_url: str, chunk_index: int) -> str:
        """
        Generate a unique chunk ID.
        
        Args:
            source_url: Source URL
            chunk_index: Chunk index
            
        Returns:
            Unique chunk ID
        """
        # Create a hash from URL and chunk index
        content = f"{source_url}_{chunk_index}_sliding_window"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    



def main():
    """Example usage of the TextChunker class."""
    
    print("=== Sliding Window Chunking Example ===")
    chunker = TextChunker(chunk_size=500, overlap=100)
    
    sample_text = """
    This is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlapThis is a sample text that will be chunked using sliding window with overlap. 
    The chunking system will split this text into manageable chunks 
    while preserving the meaning and context of the content.
    
    Each chunk will be processed independently and can be used for 
    various applications like search, analysis, or machine learning.
    
    The system uses sliding window approach with configurable overlap.
    """
    
    chunks = chunker.chunk_text(sample_text, "https://example.com")
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.text)} characters")
        print(f"Text: {chunk.text[:100]}...")
        print()


if __name__ == "__main__":
    main() 