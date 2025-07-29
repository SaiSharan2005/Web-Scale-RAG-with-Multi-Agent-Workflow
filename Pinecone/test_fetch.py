#!/usr/bin/env python3
"""
Simple test to verify fetch vectors functionality is working.
"""

import numpy as np
from Pinecone_utils import (
    initialize_pinecone, connect_to_index, create_index,
    upsert_vectors, fetch_vectors, delete_index
)

def test_fetch_vectors():
    """Test the fetch vectors functionality."""
    
    INDEX_NAME = "test-fetch-index"
    DIMENSION = 64
    
    try:
        print("🧪 Testing fetch vectors functionality...")
        
        # Initialize and create index
        initialize_pinecone()
        create_index(INDEX_NAME, DIMENSION)
        connect_to_index(INDEX_NAME)
        
        # Create test vectors
        test_vectors = [
            {
                "id": "test-vec-1",
                "values": np.random.random(DIMENSION).tolist(),
                "metadata": {"category": "test", "value": 1}
            },
            {
                "id": "test-vec-2", 
                "values": np.random.random(DIMENSION).tolist(),
                "metadata": {"category": "test", "value": 2}
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
                print(f"    Metadata: {vector_data.metadata}")
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