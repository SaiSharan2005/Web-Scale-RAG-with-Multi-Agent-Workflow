#!/usr/bin/env python3
"""
Simple test to verify fetch vectors functionality is working with Pinecone embeddings.
"""

import os
from Pinecone_utils import (
    initialize_pinecone, connect_to_index, create_index,
    upsert_vectors, fetch_vectors, delete_index,
    generate_embedding, generate_embeddings_batch,
    list_indexes
)

def test_fetch_vectors():
    """Test the fetch vectors functionality with Pinecone embeddings."""
    
    # Use environment variable or fallback to test index
    INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'test-fetch-embeddings-index')
    DIMENSION = 768  # Default dimension for BAAI/bge-base-en-v1.5 model
    
    try:
        print("🧪 Testing fetch vectors functionality...")
        
        # Initialize and create index
        initialize_pinecone()
        
        # Check if index already exists
        existing_indexes = list_indexes()
        if INDEX_NAME in existing_indexes:
            print(f"ℹ️  Index '{INDEX_NAME}' already exists, connecting to it...")
        else:
            print(f"📦 Creating new index '{INDEX_NAME}'...")
            create_index(INDEX_NAME, DIMENSION)
        
        connect_to_index(INDEX_NAME)
        
        # Create test texts and generate embeddings
        test_texts = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
            "Deep learning uses neural networks with multiple layers to process complex patterns"
        ]
        
        print("🔤 Generating embeddings for test texts...")
        embeddings = generate_embeddings_batch(test_texts)
        
        # Create test vectors with embeddings
        test_vectors = [
            {
                "id": "test-vec-1",
                "values": embeddings[0],
                "metadata": {"category": "test", "value": 1, "text": test_texts[0]}
            },
            {
                "id": "test-vec-2", 
                "values": embeddings[1],
                "metadata": {"category": "test", "value": 2, "text": test_texts[1]}
            }
        ]
        
        # Upsert vectors
        print("📤 Upserting test vectors...")
        upsert_vectors(test_vectors)
        
        # Fetch vectors
        print("📥 Fetching vectors...")
        fetch_ids = ["test-vec-1", "test-vec-2"]
        fetch_response = fetch_vectors(fetch_ids)
        
        # Check response structure
        print(f"📊 Fetch response type: {type(fetch_response)}")
        print(f"📊 Fetch response attributes: {dir(fetch_response)}")
        
        # Access vectors
        if hasattr(fetch_response, 'vectors'):
            vectors = fetch_response.vectors
            print(f"✅ Successfully accessed vectors: {len(vectors)} found")
            
            for vec_id, vector_data in vectors.items():
                print(f"  - {vec_id}: {len(vector_data.values)} dimensions")
                print(f"    Text: {vector_data.metadata.get('text', 'N/A')[:80]}...")
                print(f"    Category: {vector_data.metadata.get('category', 'N/A')}")
                print(f"    Value: {vector_data.metadata.get('value', 'N/A')}")
        else:
            print("❌ No 'vectors' attribute found in response")
            
        print("✅ Fetch vectors test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            delete_index(INDEX_NAME)
            print(f"🧹 Cleaned up test index '{INDEX_NAME}'")
        except:
            pass

if __name__ == "__main__":
    test_fetch_vectors() 