"""
Example usage of the Pinecone utilities module.

This script demonstrates:
1. Initializing the Pinecone client
2. Creating an index
3. Connecting to the index
4. Upserting vectors
5. Querying vectors
6. Fetching vectors
7. Getting index statistics
8. Cleaning up

Before running:
1. Install dependencies: pip install pinecone-client python-dotenv
2. Set your PINECONE_API_KEY in a .env file or environment variable
"""

import numpy as np
from Pinecone_utils import (
    initialize_pinecone, connect_to_index, create_index, delete_index,
    list_indexes, upsert_vectors, query_vectors, fetch_vectors,
    get_index_stats, PineconeContext
)

def main():
    """Main example function demonstrating Pinecone operations."""
    
    # Configuration
    INDEX_NAME = "example-index"
    DIMENSION = 128
    
    try:
        print("🚀 Starting Pinecone example...")
        
        # Step 1: Initialize Pinecone client
        print("\n1. Initializing Pinecone client...")
        if not initialize_pinecone():
            print("❌ Failed to initialize Pinecone client")
            return
        print("✅ Pinecone client initialized")
        
        # Step 2: List existing indexes
        print("\n2. Listing existing indexes...")
        indexes = list_indexes()
        print(f"📋 Found indexes: {indexes}")
        
        # Step 3: Create index if it doesn't exist
        print(f"\n3. Creating index '{INDEX_NAME}' if it doesn't exist...")
        if INDEX_NAME not in indexes:
            create_index(INDEX_NAME, DIMENSION, metric='cosine')
            print(f"✅ Index '{INDEX_NAME}' created")
        else:
            print(f"ℹ️  Index '{INDEX_NAME}' already exists")
        
        # Step 4: Connect to the index
        print(f"\n4. Connecting to index '{INDEX_NAME}'...")
        if not connect_to_index(INDEX_NAME):
            print("❌ Failed to connect to index")
            return
        print("✅ Connected to index")
        
        # Step 5: Generate sample vectors
        print("\n5. Generating sample vectors...")
        sample_vectors = []
        for i in range(5):
            vector_id = f"vec-{i}"
            vector_values = np.random.random(DIMENSION).tolist()
            metadata = {
                "category": f"category-{i % 3}",
                "timestamp": f"2024-01-{i+1:02d}",
                "value": i * 10
            }
            sample_vectors.append({
                "id": vector_id,
                "values": vector_values,
                "metadata": metadata
            })
        print(f"📊 Generated {len(sample_vectors)} sample vectors")
        
        # Step 6: Upsert vectors
        print("\n6. Upserting vectors to index...")
        upsert_response = upsert_vectors(sample_vectors)
        print(f"✅ Upserted vectors: {upsert_response}")
        
        # Step 7: Get index statistics
        print("\n7. Getting index statistics...")
        stats = get_index_stats()
        print(f"📈 Index stats: {stats}")
        
        # Step 8: Query vectors
        print("\n8. Querying similar vectors...")
        query_vector = np.random.random(DIMENSION).tolist()
        query_response = query_vectors(
            vector=query_vector,
            top_k=3,
            filter={"category": {"$eq": "category-1"}},
            include_metadata=True
        )
        
        print(f"🔍 Query results:")
        # New Pinecone API returns QueryResponse object, not dict
        matches = query_response.matches if hasattr(query_response, 'matches') else []
        for match in matches:
            print(f"  - ID: {match.id}, Score: {match.score:.4f}")
            print(f"    Metadata: {match.metadata if hasattr(match, 'metadata') else {}}")
        
        # Step 9: Fetch specific vectors
        print("\n9. Fetching specific vectors...")
        fetch_ids = ["vec-0", "vec-2"]
        fetch_response = fetch_vectors(fetch_ids)
        print(f"📥 Fetched vectors:")
        # New Pinecone API returns FetchResponse object, not dict
        vectors = fetch_response.vectors if hasattr(fetch_response, 'vectors') else {}
        for vec_id, vector_data in vectors.items():
            print(f"  - {vec_id}: {len(vector_data.get('values', []))} dimensions")
            print(f"    Metadata: {vector_data.get('metadata', {})}")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
    
    finally:
        # Optional: Clean up the example index
        cleanup = input("\n🗑️  Delete the example index? (y/N): ").lower().strip()
        if cleanup == 'y':
            try:
                delete_index(INDEX_NAME)
                print(f"✅ Index '{INDEX_NAME}' deleted")
            except Exception as e:
                print(f"❌ Failed to delete index: {str(e)}")

def context_manager_example():
    """Example using the context manager for automatic setup/cleanup."""
    
    print("\n🔄 Context Manager Example:")
    INDEX_NAME = "context-example"
    
    try:
        # Create index first
        initialize_pinecone()
        if INDEX_NAME not in list_indexes():
            create_index(INDEX_NAME, 128)
        
        # Use context manager
        with PineconeContext(INDEX_NAME) as ctx:
            print("📌 Inside context manager - connected to index")
            
            # Generate and upsert a simple vector
            test_vector = {
                "id": "test-vector",
                "values": np.random.random(128).tolist(),
                "metadata": {"test": True}
            }
            
            upsert_vectors([test_vector])
            print("✅ Vector upserted in context")
            
            # Query the vector
            query_result = query_vectors(
                vector=test_vector["values"],
                top_k=1,
                include_metadata=True
            )
            # New Pinecone API returns QueryResponse object, not dict
            matches_count = len(query_result.matches) if hasattr(query_result, 'matches') else 0
            print(f"🔍 Query result: {matches_count} matches")
        
        print("📌 Exited context manager")
        
        # Cleanup
        delete_index(INDEX_NAME)
        print(f"✅ Cleaned up index '{INDEX_NAME}'")
        
    except Exception as e:
        print(f"❌ Context manager example error: {str(e)}")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Run context manager example
    context_manager_example()
    
    print("\n🎉 All examples completed!")