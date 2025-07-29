# Pinecone Vector Database Utilities

A comprehensive Python wrapper for Pinecone vector database operations with environment-based configuration, proper error handling, and logging.

## 🚀 Features

- **Environment-based Configuration**: All parameters loaded from `pinecone.env`
- **Index Management**: Create, delete, list, and connect to indexes
- **Vector Operations**: Upsert, fetch, query, delete, and update vectors
- **Batch Processing**: Configurable batch sizes for efficient operations
- **Error Handling**: Comprehensive error handling with detailed logging
- **Context Management**: Automatic cleanup with context managers
- **Statistics**: Index statistics and health monitoring

## 📋 Prerequisites

1. **Pinecone Account**: Sign up at [pinecone.io](https://www.pinecone.io/)
2. **API Key**: Get your API key from the Pinecone console
3. **Python Dependencies**:
   ```bash
   pip install pinecone python-dotenv
   ```

## ⚙️ Configuration

### 1. Environment File Setup

Create a `pinecone.env` file in your project root:

```ini
# Pinecone Environment Configuration
# Copy this file to .env and modify with your actual values

# Pinecone API Configuration
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1

# Index Configuration
PINECONE_INDEX_NAME=your-index-name
PINECONE_DIMENSION=1536
PINECONE_METRIC=cosine
PINECONE_POD_TYPE=p1.x1
PINECONE_REPLICAS=1

# Query Configuration
PINECONE_TOP_K=10
PINECONE_NAMESPACE=default
PINECONE_INCLUDE_METADATA=true
PINECONE_INCLUDE_VALUES=false

# Batch Operations
PINECONE_BATCH_SIZE=100

# Logging
PINECONE_LOG_LEVEL=INFO

# Timeout Settings
PINECONE_INDEX_TIMEOUT=300
PINECONE_QUERY_TIMEOUT=30

# Export/Import Settings
PINECONE_EXPORT_FILE=pinecone_export.json
PINECONE_IMPORT_FILE=pinecone_import.json
```

### 2. Required Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `PINECONE_API_KEY` | Your Pinecone API key | - | ✅ Yes |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g., us-east-1) | - | ✅ Yes |
| `PINECONE_INDEX_NAME` | Name of your index | - | ✅ Yes |
| `PINECONE_DIMENSION` | Vector dimension | 1536 | ❌ No |
| `PINECONE_METRIC` | Distance metric (cosine, euclidean, dotproduct) | cosine | ❌ No |

### 3. Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `PINECONE_TOP_K` | Number of results for queries | 10 |
| `PINECONE_NAMESPACE` | Default namespace | default |
| `PINECONE_BATCH_SIZE` | Batch size for operations | 100 |
| `PINECONE_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PINECONE_INCLUDE_METADATA` | Include metadata in query results | true |
| `PINECONE_INCLUDE_VALUES` | Include vector values in query results | false |

## 🔧 Installation

1. **Clone or download** the Pinecone utils module
2. **Install dependencies**:
   ```bash
   pip install pinecone python-dotenv
   ```
3. **Create environment file**:
   ```bash
   cp pinecone.env.example pinecone.env
   # Edit pinecone.env with your actual values
   ```

## 📖 Usage

### Basic Setup

```python
from Pinecone.Pinecone_utils import initialize_pinecone, connect_to_index

# Initialize Pinecone (loads from pinecone.env)
initialize_pinecone()

# Connect to your index
connect_to_index()
```

### Index Management

```python
from Pinecone.Pinecone_utils import create_index, delete_index, list_indexes

# Create a new index
create_index()  # Uses parameters from pinecone.env

# List all indexes
indexes = list_indexes()
print(f"Available indexes: {indexes}")

# Delete an index
delete_index("old-index-name")
```

### Vector Operations

```python
from Pinecone.Pinecone_utils import upsert_vectors, query_vectors, fetch_vectors

# Prepare vectors for upserting
vectors = [
    {
        "id": "vec1",
        "values": [0.1, 0.2, 0.3, ...],  # Your vector values
        "metadata": {"text": "Sample text", "category": "example"}
    },
    {
        "id": "vec2", 
        "values": [0.4, 0.5, 0.6, ...],
        "metadata": {"text": "Another sample", "category": "example"}
    }
]

# Upsert vectors
response = upsert_vectors(vectors)
print(f"Upserted {response['upserted_count']} vectors")

# Query similar vectors
query_vector = [0.1, 0.2, 0.3, ...]  # Your query vector
results = query_vectors(query_vector, top_k=5)
print(f"Found {len(results.matches)} similar vectors")

# Fetch specific vectors
fetched = fetch_vectors(["vec1", "vec2"])
print(f"Fetched {len(fetched.vectors)} vectors")
```

### Advanced Usage

```python
from Pinecone.Pinecone_utils import PineconeContext, get_index_stats

# Using context manager for automatic cleanup
with PineconeContext() as pc:
    # All operations within this block
    stats = get_index_stats()
    print(f"Index has {stats['total_vector_count']} vectors")

# Query with filters
results = query_vectors(
    vector=query_vector,
    top_k=10,
    filter={"category": "example"},
    include_metadata=True
)

# Update a specific vector
from Pinecone.Pinecone_utils import update_vector
update_vector(
    id="vec1",
    vector=new_vector_values,
    metadata={"updated": True}
)
```

## 🔍 API Reference

### Core Functions

#### `initialize_pinecone(api_key=None, environment=None)`
Initialize the Pinecone client.
- **Returns**: `bool` - True if successful

#### `connect_to_index(index_name=None)`
Connect to a specific Pinecone index.
- **Returns**: `bool` - True if successful

#### `create_index(name=None, dimension=None, metric=None, cloud='aws', region=None)`
Create a new Pinecone index.
- **Raises**: `Exception` if creation fails

#### `delete_index(name)`
Delete a Pinecone index.
- **Raises**: `Exception` if deletion fails

#### `list_indexes()`
List all available indexes.
- **Returns**: `List[str]` - List of index names

### Vector Operations

#### `upsert_vectors(vectors, batch_size=None)`
Insert or update vectors in batches.
- **Args**: 
  - `vectors`: List of vector dictionaries
  - `batch_size`: Optional batch size (defaults to env var)
- **Returns**: `Dict` - Upsert response

#### `query_vectors(vector, top_k=None, filter=None, include_metadata=None, include_values=None)`
Query vectors by similarity.
- **Args**:
  - `vector`: Query vector as list of floats
  - `top_k`: Number of results (defaults to env var)
  - `filter`: Optional metadata filter
  - `include_metadata`: Include metadata in results
  - `include_values`: Include vector values in results
- **Returns**: `QueryResponse` - Query results

#### `fetch_vectors(ids)`
Fetch vectors by their IDs.
- **Args**: `ids` - List of vector IDs
- **Returns**: `FetchResponse` - Fetched vectors

#### `delete_vectors(ids)`
Delete vectors by their IDs.
- **Args**: `ids` - List of vector IDs
- **Returns**: `Dict` - Delete response

### Utility Functions

#### `update_vector(id, vector, metadata=None)`
Update a single vector.
- **Args**:
  - `id`: Vector ID
  - `vector`: New vector values
  - `metadata`: Optional metadata

#### `get_index_stats()`
Get index statistics.
- **Returns**: `Dict` - Index statistics

#### `delete_all_vectors(namespace="")`
Delete all vectors in index or namespace.
- **Args**: `namespace` - Optional namespace (empty for default)

#### `get_current_index_name()`
Get current index name.
- **Returns**: `Optional[str]` - Current index name

#### `is_connected()`
Check if client is connected.
- **Returns**: `bool` - True if connected

### Context Manager

#### `PineconeContext(index_name=None, api_key=None, environment=None)`
Context manager for automatic cleanup.

```python
with PineconeContext() as pc:
    # Your Pinecone operations here
    pass  # Automatic cleanup on exit
```

## 📊 Examples

### Complete Workflow Example

```python
from Pinecone.Pinecone_utils import *

def complete_workflow():
    # 1. Initialize and connect
    initialize_pinecone()
    connect_to_index()
    
    # 2. Create sample vectors
    vectors = [
        {
            "id": "doc1",
            "values": [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
            "metadata": {"title": "Document 1", "category": "tech"}
        },
        {
            "id": "doc2", 
            "values": [0.4, 0.5, 0.6] * 512,
            "metadata": {"title": "Document 2", "category": "science"}
        }
    ]
    
    # 3. Upsert vectors
    upsert_response = upsert_vectors(vectors)
    print(f"Upserted {upsert_response['upserted_count']} vectors")
    
    # 4. Query similar vectors
    query_vec = [0.1, 0.2, 0.3] * 512
    results = query_vectors(
        vector=query_vec,
        top_k=5,
        filter={"category": "tech"},
        include_metadata=True
    )
    
    # 5. Process results
    for match in results.matches:
        print(f"ID: {match.id}, Score: {match.score}")
        if match.metadata:
            print(f"Title: {match.metadata.get('title', 'N/A')}")
    
    # 6. Get index statistics
    stats = get_index_stats()
    print(f"Total vectors: {stats['total_vector_count']}")

if __name__ == "__main__":
    complete_workflow()
```

### Error Handling Example

```python
from Pinecone.Pinecone_utils import *

def safe_operations():
    try:
        # Initialize
        if not initialize_pinecone():
            print("Failed to initialize Pinecone")
            return
        
        # Connect to index
        if not connect_to_index():
            print("Failed to connect to index")
            return
        
        # Perform operations
        vectors = [{"id": "test", "values": [0.1] * 1536}]
        upsert_vectors(vectors)
        
    except Exception as e:
        print(f"Error: {e}")
        # Handle specific error types
        if "API key" in str(e):
            print("Check your PINECONE_API_KEY in pinecone.env")
        elif "index" in str(e):
            print("Check your PINECONE_INDEX_NAME in pinecone.env")
```

## 🐛 Troubleshooting

### Common Issues

1. **"Invalid API Key" Error**
   - Verify your `PINECONE_API_KEY` in `pinecone.env`
   - Check that the key is active in your Pinecone console

2. **"Index not found" Error**
   - Verify your `PINECONE_INDEX_NAME` in `pinecone.env`
   - Use `list_indexes()` to see available indexes

3. **"Environment not found" Error**
   - Check your `PINECONE_ENVIRONMENT` setting
   - Common environments: `us-east-1`, `us-west1-gcp`, `eu-west1-aws`

4. **"Dimension mismatch" Error**
   - Ensure your vectors match the index dimension
   - Check `PINECONE_DIMENSION` in your environment file

### Debug Mode

Enable debug logging by setting in `pinecone.env`:
```ini
PINECONE_LOG_LEVEL=DEBUG
```

### Health Check

```python
from Pinecone.Pinecone_utils import is_connected, get_current_index_name

if is_connected():
    print(f"Connected to index: {get_current_index_name()}")
else:
    print("Not connected to any index")
```

## 📝 Best Practices

1. **Environment Variables**: Always use `pinecone.env` for configuration
2. **Batch Operations**: Use appropriate batch sizes for large datasets
3. **Error Handling**: Wrap operations in try-catch blocks
4. **Context Managers**: Use `PineconeContext` for automatic cleanup
5. **Logging**: Monitor logs for debugging and monitoring
6. **Vector Dimensions**: Ensure consistency across your application
7. **Metadata**: Use meaningful metadata for filtering and organization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Pinecone documentation at [docs.pinecone.io](https://docs.pinecone.io/)
3. Open an issue in the repository

---

**Note**: This utility is designed to work with the latest Pinecone Python client. Make sure you have the latest version installed: `pip install --upgrade pinecone` 