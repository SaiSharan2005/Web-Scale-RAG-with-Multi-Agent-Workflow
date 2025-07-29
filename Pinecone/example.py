#!/usr/bin/env python3
"""
Pinecone Utils Example

This example demonstrates how to use the Pinecone utilities with text embeddings
for semantic search and document retrieval.

Features demonstrated:
- Initialize Pinecone with environment configuration
- Generate text embeddings using SentenceTransformer
- Upsert vectors with metadata
- Query similar documents
- Handle errors gracefully

Usage:
    python example.py
"""

import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the path to import Pinecone_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pinecone.Pinecone_utils import (
    initialize_pinecone, 
    connect_to_index,
    generate_embedding,
    generate_embeddings_batch,
    upsert_vectors,
    query_vectors,
    fetch_vectors,
    get_index_stats,
    list_indexes,
    is_connected,
    get_current_index_name
)

def setup_pinecone() -> bool:
    """
    Initialize and connect to Pinecone.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        print("🔧 Initializing Pinecone...")
        
        # Initialize Pinecone (loads from pinecone.env)
        if not initialize_pinecone():
            print("❌ Failed to initialize Pinecone")
            return False
        
        # Connect to index (loads from pinecone.env)
        if not connect_to_index():
            print("❌ Failed to connect to index")
            return False
        
        print(f"✅ Connected to index: {get_current_index_name()}")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def demonstrate_embeddings():
    """Demonstrate text embedding generation."""
    print("\n🔤 Text Embedding Examples")
    print("=" * 50)
    
    # Single text embedding
    sample_text = "Machine learning is a subset of artificial intelligence"
    print(f"Generating embedding for: '{sample_text}'")
    
    try:
        embedding = generate_embedding(sample_text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return
    
    # Batch embeddings
    texts = [
        "Deep learning uses neural networks",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to see and interpret images",
        "Reinforcement learning learns through trial and error"
    ]
    
    print(f"\nGenerating batch embeddings for {len(texts)} texts...")
    
    try:
        embeddings = generate_embeddings_batch(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            print(f"   Text {i+1}: {len(emb)} dimensions")
            
    except Exception as e:
        print(f"❌ Batch embedding generation failed: {e}")

def demonstrate_vector_operations():
    """Demonstrate vector operations with embeddings."""
    print("\n📊 Vector Operations Examples")
    print("=" * 50)
    
    # Sample documents with metadata
    documents = [
        {
            "id": "doc1",
            "text": "Machine learning algorithms can learn patterns from data",
            "metadata": {
                "title": "ML Introduction",
                "category": "technology",
                "author": "AI Expert"
            }
        },
        {
            "id": "doc2",
            "text": "Deep learning uses neural networks with multiple layers for complex tasks",
            "metadata": {
                "title": "Deep Learning Basics",
                "category": "technology", 
                "author": "ML Researcher"
            }
        },
        {
            "id": "doc3",
            "text": "Natural language processing enables computers to understand human language",
            "metadata": {
                "title": "NLP Overview",
                "category": "technology",
                "author": "NLP Specialist"
            }
        },
        {
            "id": "doc4",
            "text": "Computer vision helps machines interpret and analyze visual information",
            "metadata": {
                "title": "Computer Vision",
                "category": "technology",
                "author": "CV Engineer"
            }
        },
        {
            "id": "doc5",
            "text": "Reinforcement learning agents learn optimal behavior through interaction",
            "metadata": {
                "title": "Reinforcement Learning",
                "category": "technology",
                "author": "RL Scientist"
            }
        }
    ]
    
    print(f"Preparing {len(documents)} documents for vector operations...")
    
    try:
        # Generate embeddings for all documents
        texts = [doc["text"] for doc in documents]
        embeddings = generate_embeddings_batch(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, doc in enumerate(documents):
            vectors.append({
                "id": doc["id"],
                "values": embeddings[i],
                "metadata": {
                    "text": doc["text"],
                    "title": doc["metadata"]["title"],
                    "category": doc["metadata"]["category"],
                    "author": doc["metadata"]["author"]
                }
            })
        
        print(f"✅ Prepared {len(vectors)} vectors with embeddings")
        
        # Upsert vectors to Pinecone
        print("\n📤 Upserting vectors to Pinecone...")
        upsert_response = upsert_vectors(vectors)
        print(f"✅ Upserted {upsert_response['upserted_count']} vectors successfully")
        
        # Get index statistics
        stats = get_index_stats()
        print(f"📈 Index statistics: {stats['total_vector_count']} total vectors")
        
        return vectors
        
    except Exception as e:
        print(f"❌ Vector operations failed: {e}")
        return None

def demonstrate_search():
    """Demonstrate semantic search capabilities."""
    print("\n🔍 Semantic Search Examples")
    print("=" * 50)
    
    # Sample queries
    queries = [
        "How do machines learn from data?",
        "What is neural network technology?",
        "How can computers understand language?",
        "What is visual computing?",
        "How do agents learn through experience?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔎 Query {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Generate embedding for query
            query_embedding = generate_embedding(query)
            
            # Search for similar documents
            results = query_vectors(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            print(f"Found {len(results.matches)} similar documents:")
            
            for j, match in enumerate(results.matches, 1):
                print(f"  {j}. Score: {match.score:.4f}")
                print(f"     Title: {match.metadata.get('title', 'N/A')}")
                print(f"     Author: {match.metadata.get('author', 'N/A')}")
                print(f"     Text: {match.metadata.get('text', 'N/A')[:80]}...")
                print()
                
        except Exception as e:
            print(f"❌ Search failed: {e}")

def demonstrate_filtered_search():
    """Demonstrate search with metadata filters."""
    print("\n🎯 Filtered Search Examples")
    print("=" * 50)
    
    query = "How do machines learn?"
    print(f"🔎 Query: '{query}'")
    
    try:
        query_embedding = generate_embedding(query)
        
        # Search with category filter
        print("\n📂 Filtering by category 'technology':")
        results = query_vectors(
            vector=query_embedding,
            top_k=5,
            filter={"category": "technology"},
            include_metadata=True
        )
        
        print(f"Found {len(results.matches)} technology documents:")
        for i, match in enumerate(results.matches, 1):
            print(f"  {i}. {match.metadata.get('title', 'N/A')} (Score: {match.score:.4f})")
        
        # Search with author filter
        print("\n👤 Filtering by author 'AI Expert':")
        results = query_vectors(
            vector=query_embedding,
            top_k=5,
            filter={"author": "AI Expert"},
            include_metadata=True
        )
        
        print(f"Found {len(results.matches)} documents by AI Expert:")
        for i, match in enumerate(results.matches, 1):
            print(f"  {i}. {match.metadata.get('title', 'N/A')} (Score: {match.score:.4f})")
            
    except Exception as e:
        print(f"❌ Filtered search failed: {e}")

def demonstrate_fetch():
    """Demonstrate fetching specific vectors."""
    print("\n📥 Fetch Examples")
    print("=" * 50)
    
    # Fetch specific documents
    doc_ids = ["doc1", "doc3", "doc5"]
    print(f"Fetching documents: {doc_ids}")
    
    try:
        fetch_response = fetch_vectors(doc_ids)
        print(f"✅ Fetched {len(fetch_response.vectors)} vectors")
        
        for doc_id in doc_ids:
            if doc_id in fetch_response.vectors:
                vector = fetch_response.vectors[doc_id]
                print(f"  📄 {doc_id}: {len(vector.values)} dimensions")
                if vector.metadata:
                    print(f"     Title: {vector.metadata.get('title', 'N/A')}")
            else:
                print(f"  ❌ {doc_id}: Not found")
                
    except Exception as e:
        print(f"❌ Fetch failed: {e}")

def main():
    """Main example function."""
    print("🚀 Pinecone Utils Example")
    print("=" * 60)
    print("This example demonstrates Pinecone utilities with text embeddings")
    print("Make sure you have configured pinecone.env with your credentials")
    print()
    
    # Check if connected
    if not is_connected():
        print("❌ Not connected to Pinecone. Please check your configuration.")
        print("   Ensure pinecone.env is properly configured with:")
        print("   - PINECONE_API_KEY")
        print("   - PINECONE_ENVIRONMENT") 
        print("   - PINECONE_INDEX_NAME")
        return
    
    # Setup Pinecone
    if not setup_pinecone():
        return
    
    # List available indexes
    try:
        indexes = list_indexes()
        print(f"📋 Available indexes: {indexes}")
    except Exception as e:
        print(f"⚠️  Could not list indexes: {e}")
    
    # Demonstrate embeddings
    demonstrate_embeddings()
    
    # Demonstrate vector operations
    vectors = demonstrate_vector_operations()
    if not vectors:
        return
    
    # Demonstrate search
    demonstrate_search()
    
    # Demonstrate filtered search
    demonstrate_filtered_search()
    
    # Demonstrate fetch
    demonstrate_fetch()
    
    print("\n✅ Example completed successfully!")
    print("\n💡 Tips:")
    print("   - Check the logs for detailed information")
    print("   - Use different embedding models for specific use cases")
    print("   - Experiment with different metadata filters")
    print("   - Monitor index statistics for performance")

if __name__ == "__main__":
    main()