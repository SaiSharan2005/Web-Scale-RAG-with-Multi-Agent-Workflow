#!/usr/bin/env python3
"""
Website RAG Pipeline

A complete pipeline that:
1. Crawls websites using the DomainCrawler
2. Chunks content using TextChunker
3. Stores chunks in Pinecone with website-specific filtering
4. Integrates with the RAG system for website-specific queries

Usage:
    python website_rag_pipeline.py --website https://example.com --name "Example Website"
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Add paths for imports
sys.path.append('../Crawler')
sys.path.append('../Chunking')
sys.path.append('../Pinecone')

from crawler import DomainCrawler
from chunking import TextChunker, ChunkingConfig
from Pinecone_utils import initialize_pinecone, create_index, connect_to_index, upsert_vectors, query_vectors, generate_embedding

# Load environment variables
try:
    from dotenv import load_dotenv
    # Load Pinecone configuration from pinecone.env (as per README)
    load_dotenv('../Pinecone/pinecone.env')
    # Load other API keys from config.env
    load_dotenv('config.env')
    print("Successfully loaded API keys from config.env and pinecone.env")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")

class WebsiteRAGPipeline:
    def __init__(self, website_url: str, website_name: str, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the Website RAG Pipeline.
        
        Args:
            website_url: URL of the website to crawl
            website_name: Name/identifier for the website
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        self.website_url = website_url
        self.website_name = website_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.crawler = None
        self.chunker = None
        self.pinecone_initialized = False
        
    def initialize_pinecone(self) -> bool:
        """Initialize Pinecone connection using Pinecone utilities."""
        try:
            # Initialize Pinecone (loads from pinecone.env as per README)
            if not initialize_pinecone():
                self.logger.error("Failed to initialize Pinecone")
                return False
            try:
                index_name = os.getenv('PINECONE_INDEX_NAME', 'rag-model-agentic-ai')
                create_index(index_name)
                self.logger.info(f"Created new Pinecone index: {index_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    self.logger.info(f"Pinecone index already exists: {index_name}")
                else:
                    self.logger.error(f"Error creating index: {e}")
                    return False
            # Connect to index (uses PINECONE_INDEX_NAME from pinecone.env)
            if not connect_to_index():
                self.logger.error("Failed to connect to Pinecone index")
                return False
            
            self.pinecone_initialized = True
            self.logger.info("Pinecone initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Pinecone: {e}")
            return False
    
    def crawl_website(self, delay: float = 1.0, limit: Optional[int] = None) -> Dict[str, str]:
        """
        Crawl the website using DomainCrawler.
        
        Args:
            delay: Delay between requests
            limit: Maximum number of pages to crawl
            
        Returns:
            Dictionary mapping URLs to text content
        """
        self.logger.info(f"Starting crawl of website: {self.website_url}")
        
        try:
            self.crawler = DomainCrawler(
                start_url=self.website_url,
                delay=delay,
                limit=limit,
                respect_robots=True
            )
            
            results = self.crawler.crawl()
            self.logger.info(f"Crawling complete. Extracted content from {len(results)} pages")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during crawling: {e}")
            return {}
    
    def chunk_content(self, crawled_data: Dict[str, str]) -> List:
        """
        Chunk the crawled content using TextChunker.
        
        Args:
            crawled_data: Dictionary mapping URLs to text content
            
        Returns:
            List of TextChunk objects
        """
        self.logger.info("Starting content chunking")
        
        try:
            config = ChunkingConfig(
                chunk_size=self.chunk_size,
                overlap_size=self.overlap,
                clean_text=True,
                normalize_whitespace=True
            )
            
            self.chunker = TextChunker(config)
            chunks = self.chunker.process_crawled_data(crawled_data)
            
            self.logger.info(f"Chunking complete. Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error during chunking: {e}")
            return []
    
    def prepare_vectors_for_pinecone(self, chunks: List) -> List[Dict]:
        """
        Prepare chunks for Pinecone storage with website-specific metadata.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of vector dictionaries for Pinecone
        """
        vectors = []
        
        for chunk in chunks:
            # Create metadata with website-specific information
            metadata = {
                'text': chunk.text,
                'website_name': self.website_name,
                'website_url': self.website_url,
                'source_url': chunk.source_url,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'chunk_size': chunk.chunk_size,
                'overlap_size': chunk.overlap_size,
                'created_at': chunk.created_at,
                'processed_at': datetime.now().isoformat()
            }
            
            # Create vector data
            vector_data = {
                'id': chunk.id,
                'values': self._generate_embedding(chunk.text),  # You'll need to implement this
                'metadata': metadata
            }
            
            vectors.append(vector_data)
        
        return vectors
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Pinecone utilities.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        try:
            # Use the Pinecone utility function
            embedding = generate_embedding(text)
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return a dummy embedding (you should handle this properly)
            return [0.0] * 768  # BGE model dimension
    def store_in_pinecone(self, vectors: List[Dict]) -> bool:
        """
        Store vectors in Pinecone with website-specific filtering.
        
        Args:
            vectors: List of vector dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pinecone_initialized:
            self.logger.error("Pinecone not initialized")
            return False
        
        try:
            # Store vectors in batches
            batch_size = 100
            total_stored = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                response = upsert_vectors(batch, batch_size=batch_size)
                total_stored += response.get('upserted_count', len(batch))
                
                self.logger.info(f"Stored batch {i//batch_size + 1}: {len(batch)} vectors")
            
            self.logger.info(f"Successfully stored {total_stored} vectors in Pinecone")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing in Pinecone: {e}")
            return False
    
    def query_website_specific(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query Pinecone for website-specific content.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.pinecone_initialized:
            self.logger.error("Pinecone not initialized")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self._generate_embedding(query)
            
            # Query with website-specific filter
            filter_dict = {
                'website_name': {'$eq': self.website_name}
            }
            
            results = query_vectors(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True,
                include_values=False
            )
            
            # Extract and format results
            documents = []
            if hasattr(results, 'matches'):
                for match in results.matches:
                    if hasattr(match, 'metadata') and match.metadata:
                        documents.append({
                            'text': match.metadata.get('text', ''),
                            'source_url': match.metadata.get('source_url', ''),
                            'score': getattr(match, 'score', 0.0),
                            'website_name': match.metadata.get('website_name', ''),
                            'chunk_index': match.metadata.get('chunk_index', 0)
                        })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error querying Pinecone: {e}")
            return []
    
    def run_pipeline(self, delay: float = 1.0, limit: Optional[int] = None) -> bool:
        """
        Run the complete pipeline: crawl -> chunk -> store.
        
        Args:
            delay: Delay between crawler requests
            limit: Maximum pages to crawl
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Starting Website RAG Pipeline for: {self.website_name}")
        
        # Step 1: Initialize Pinecone
        if not self.initialize_pinecone():
            return False
        
        # Step 2: Crawl website
        crawled_data = self.crawl_website(delay=delay, limit=limit)
        if not crawled_data:
            self.logger.error("No data crawled")
            return False
        
        # Step 3: Chunk content
        chunks = self.chunk_content(crawled_data)
        if not chunks:
            self.logger.error("No chunks created")
            return False
        
        # Step 4: Prepare vectors
        vectors = self.prepare_vectors_for_pinecone(chunks)
        if not vectors:
            self.logger.error("No vectors prepared")
            return False
        
        # Step 5: Store in Pinecone
        if not self.store_in_pinecone(vectors):
            return False
        
        self.logger.info(f"Pipeline completed successfully for {self.website_name}")
        return True


def main():
    """Main function to run the pipeline from command line."""
    parser = argparse.ArgumentParser(description='Website RAG Pipeline')
    parser.add_argument('--website', required=True, help='Website URL to crawl')
    parser.add_argument('--name', required=True, help='Website name/identifier')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size (default: 1000)')
    parser.add_argument('--overlap', type=int, default=200, help='Chunk overlap (default: 200)')
    parser.add_argument('--delay', type=float, default=1.0, help='Crawler delay (default: 1.0)')
    parser.add_argument('--limit', type=int, help='Maximum pages to crawl')
    parser.add_argument('--query', help='Test query after pipeline completion')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = WebsiteRAGPipeline(
        website_url=args.website,
        website_name=args.name,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Run the pipeline
    success = pipeline.run_pipeline(delay=args.delay, limit=args.limit)
    
    if success:
        print(f"\n✅ Pipeline completed successfully for {args.name}")
        
        # Test query if provided
        if args.query:
            print(f"\n🔍 Testing query: {args.query}")
            results = pipeline.query_website_specific(args.query)
            
            if results:
                print(f"Found {len(results)} relevant documents:")
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. Score: {doc['score']:.3f}")
                    print(f"   Source: {doc['source_url']}")
                    print(f"   Text: {doc['text'][:200]}...")
            else:
                print("No relevant documents found")
    else:
        print(f"\n❌ Pipeline failed for {args.name}")


if __name__ == "__main__":
    main() 