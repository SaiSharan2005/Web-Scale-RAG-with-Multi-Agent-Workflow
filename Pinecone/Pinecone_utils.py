"""
Pinecone Vector Database Utility Module

A comprehensive wrapper for common Pinecone operations including index management,
vector operations, and utility functions with proper error handling and logging.

Dependencies:
    pip install pinecone-client python-dotenv

Environment Variables:
    PINECONE_API_KEY: Your Pinecone API key
    PINECONE_ENVIRONMENT: Your Pinecone environment (optional)
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec, PodSpec

# Load environment variables from pinecone.env
load_dotenv('pinecone.env')

# Configure logging
log_level = os.getenv('PINECONE_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

class PineconeClient:
    """Pinecone client wrapper with connection management."""
    
    def __init__(self):
        self.pc = None
        self.current_index = None
        self.index_name = None
    
    def initialize(self, api_key: Optional[str] = None, environment: Optional[str] = None) -> bool:
        """
        Initialize the Pinecone client.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load all parameters from environment
            api_key = api_key or os.getenv('PINECONE_API_KEY')
            environment = environment or os.getenv('PINECONE_ENVIRONMENT')
            
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables or parameters")
            
            self.pc = Pinecone(api_key=api_key)
            logger.info(f"Pinecone client initialized successfully with environment: {environment}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            return False
    
    def connect_to_index(self, index_name: str) -> bool:
        """
        Connect to a specific Pinecone index.
        
        Args:
            index_name: Name of the index to connect to
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if not self.pc:
                raise ValueError("Pinecone client not initialized. Call initialize() first.")
            
            if index_name not in [idx.name for idx in self.pc.list_indexes()]:
                raise ValueError(f"Index '{index_name}' does not exist")
            
            self.current_index = self.pc.Index(index_name)
            self.index_name = index_name
            logger.info(f"Connected to index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to index '{index_name}': {str(e)}")
            return False

# Global client instance
_client = PineconeClient()

def initialize_pinecone(api_key: Optional[str] = None, environment: Optional[str] = None) -> bool:
    """
    Initialize the Pinecone client with API credentials.
    
    Args:
        api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
        environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    return _client.initialize(api_key, environment)

def connect_to_index(index_name: str = None) -> bool:
    """
    Connect to a specific Pinecone index.
    
    Args:
        index_name: Name of the index to connect to (defaults to PINECONE_INDEX_NAME env var)
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    index_name = index_name or os.getenv('PINECONE_INDEX_NAME')
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in environment variables or parameters")
    
    return _client.connect_to_index(index_name)

# Index Management Functions

def create_index(name: str = None, dimension: int = None, metric: str = None, 
                cloud: str = 'aws', region: str = None) -> None:
    """
    Create a new Pinecone index.
    
    Args:
        name: Name of the index to create (defaults to PINECONE_INDEX_NAME env var)
        dimension: Vector dimension for the index (defaults to PINECONE_DIMENSION env var)
        metric: Distance metric ('cosine', 'euclidean', 'dotproduct') (defaults to PINECONE_METRIC env var)
        cloud: Cloud provider ('aws', 'gcp', 'azure')
        region: Cloud region (defaults to PINECONE_ENVIRONMENT env var)
        
    Raises:
        Exception: If index creation fails
    """
    try:
        if not _client.pc:
            raise ValueError("Pinecone client not initialized. Call initialize_pinecone() first.")
        
        # Load parameters from environment
        name = name or os.getenv('PINECONE_INDEX_NAME')
        dimension = dimension or int(os.getenv('PINECONE_DIMENSION', 1536))
        metric = metric or os.getenv('PINECONE_METRIC', 'cosine')
        region = region or os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        
        if not name:
            raise ValueError("PINECONE_INDEX_NAME not found in environment variables or parameters")
        
        spec = ServerlessSpec(cloud=cloud, region=region)
        
        _client.pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=spec
        )
        logger.info(f"Index '{name}' created successfully with dimension={dimension}, metric={metric}")
        
    except Exception as e:
        error_msg = f"Failed to create index '{name}': {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def delete_index(name: str) -> None:
    """
    Delete a Pinecone index.
    
    Args:
        name: Name of the index to delete
        
    Raises:
        Exception: If index deletion fails
    """
    try:
        if not _client.pc:
            raise ValueError("Pinecone client not initialized. Call initialize_pinecone() first.")
        
        _client.pc.delete_index(name)
        logger.info(f"Index '{name}' deleted successfully")
        
        # Reset current index if it was the deleted one
        if _client.index_name == name:
            _client.current_index = None
            _client.index_name = None
            
    except Exception as e:
        error_msg = f"Failed to delete index '{name}': {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def list_indexes() -> List[str]:
    """
    List all available Pinecone indexes.
    
    Returns:
        List[str]: List of index names
        
    Raises:
        Exception: If listing indexes fails
    """
    try:
        if not _client.pc:
            raise ValueError("Pinecone client not initialized. Call initialize_pinecone() first.")
        
        indexes = [idx.name for idx in _client.pc.list_indexes()]
        logger.info(f"Found {len(indexes)} indexes")
        return indexes
        
    except Exception as e:
        error_msg = f"Failed to list indexes: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

# Data Operations Functions

def upsert_vectors(vectors: List[Dict], batch_size: int = None) -> Dict:
    """
    Insert or update vectors in the current index.
    
    Args:
        vectors: List of vector dictionaries with 'id', 'values', and optional 'metadata'
        batch_size: Number of vectors to process in each batch (defaults to PINECONE_BATCH_SIZE env var)
        
    Returns:
        Dict: Upsert response from Pinecone
        
    Raises:
        Exception: If upsert operation fails
    """
    try:
        if not _client.current_index:
            raise ValueError("No index connected. Call connect_to_index() first.")
        
        # Load batch size from environment
        batch_size = batch_size or int(os.getenv('PINECONE_BATCH_SIZE', 100))
        
        # Process vectors in batches
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            response = _client.current_index.upsert(vectors=batch)
            total_upserted += response.get('upserted_count', len(batch))
            logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
        
        logger.info(f"Total upserted {total_upserted} vectors successfully")
        return {'upserted_count': total_upserted}
        
    except Exception as e:
        error_msg = f"Failed to upsert vectors: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def fetch_vectors(ids: List[str]) -> Any:
    """
    Fetch vectors by their IDs.
    
    Args:
        ids: List of vector IDs to fetch
        
    Returns:
        FetchResponse: Fetch response containing vectors
        
    Raises:
        Exception: If fetch operation fails
    """
    try:
        if not _client.current_index:
            raise ValueError("No index connected. Call connect_to_index() first.")
        
        response = _client.current_index.fetch(ids=ids)
        logger.info(f"Fetched {len(ids)} vectors successfully")
        return response
        
    except Exception as e:
        error_msg = f"Failed to fetch vectors: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def query_vectors(vector: List[float], top_k: int = None, 
                 filter: Optional[Dict] = None, include_metadata: bool = None,
                 include_values: bool = None) -> Any:
    """
    Query vectors by similarity to a query vector.
    
    Args:
        vector: Query vector as list of floats
        top_k: Number of most similar vectors to return (defaults to PINECONE_TOP_K env var)
        filter: Optional metadata filter
        include_metadata: Whether to include metadata in results (defaults to PINECONE_INCLUDE_METADATA env var)
        include_values: Whether to include vector values in results (defaults to PINECONE_INCLUDE_VALUES env var)
        
    Returns:
        QueryResponse: Query response with similar vectors
        
    Raises:
        Exception: If query operation fails
    """
    try:
        if not _client.current_index:
            raise ValueError("No index connected. Call connect_to_index() first.")
        
        # Load parameters from environment
        top_k = top_k or int(os.getenv('PINECONE_TOP_K', 10))
        include_metadata = include_metadata if include_metadata is not None else os.getenv('PINECONE_INCLUDE_METADATA', 'true').lower() == 'true'
        include_values = include_values if include_values is not None else os.getenv('PINECONE_INCLUDE_VALUES', 'false').lower() == 'true'
        
        response = _client.current_index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
            include_values=include_values
        )
        # New API returns QueryResponse object, not dict
        matches_count = len(response.matches) if hasattr(response, 'matches') else 0
        logger.info(f"Query completed, returned {matches_count} results")
        return response
        
    except Exception as e:
        error_msg = f"Failed to query vectors: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def delete_vectors(ids: List[str]) -> Dict:
    """
    Delete vectors by their IDs.
    
    Args:
        ids: List of vector IDs to delete
        
    Returns:
        Dict: Delete response from Pinecone
        
    Raises:
        Exception: If delete operation fails
    """
    try:
        if not _client.current_index:
            raise ValueError("No index connected. Call connect_to_index() first.")
        
        response = _client.current_index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors successfully")
        return response
        
    except Exception as e:
        error_msg = f"Failed to delete vectors: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

# Utility Functions

def update_vector(id: str, vector: List[float], metadata: Optional[Dict] = None) -> Dict:
    """
    Update a single vector by ID.
    
    Args:
        id: Vector ID to update
        vector: New vector values
        metadata: Optional metadata to update
        
    Returns:
        Dict: Upsert response from Pinecone
        
    Raises:
        Exception: If update operation fails
    """
    try:
        if not _client.current_index:
            raise ValueError("No index connected. Call connect_to_index() first.")
        
        vector_data = {"id": id, "values": vector}
        if metadata:
            vector_data["metadata"] = metadata
            
        response = _client.current_index.upsert(vectors=[vector_data])
        logger.info(f"Updated vector '{id}' successfully")
        return response
        
    except Exception as e:
        error_msg = f"Failed to update vector '{id}': {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def get_index_stats() -> Dict:
    """
    Get statistics about the current index.
    
    Returns:
        Dict: Index statistics including vector count, dimension, etc.
        
    Raises:
        Exception: If stats retrieval fails
    """
    try:
        if not _client.current_index:
            raise ValueError("No index connected. Call connect_to_index() first.")
        
        stats = _client.current_index.describe_index_stats()
        logger.info(f"Retrieved stats for index '{_client.index_name}'")
        return stats
        
    except Exception as e:
        error_msg = f"Failed to get index stats: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def delete_all_vectors(namespace: str = "") -> Dict:
    """
    Delete all vectors in the current index or namespace.
    
    Args:
        namespace: Optional namespace to delete from (empty string for default)
        
    Returns:
        Dict: Delete response from Pinecone
        
    Raises:
        Exception: If delete operation fails
    """
    try:
        if not _client.current_index:
            raise ValueError("No index connected. Call connect_to_index() first.")
        
        response = _client.current_index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted all vectors from index '{_client.index_name}'")
        return response
        
    except Exception as e:
        error_msg = f"Failed to delete all vectors: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

# Context manager for automatic cleanup
class PineconeContext:
    """Context manager for Pinecone operations with automatic cleanup."""
    
    def __init__(self, index_name: str = None, api_key: Optional[str] = None, 
                 environment: Optional[str] = None):
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME')
        self.api_key = api_key
        self.environment = environment
    
    def __enter__(self):
        initialize_pinecone(self.api_key, self.environment)
        connect_to_index(self.index_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Could add cleanup logic here if needed
        pass

def get_current_index_name() -> Optional[str]:
    """
    Get the name of the currently connected index.
    
    Returns:
        Optional[str]: Current index name or None if not connected
    """
    return _client.index_name

def is_connected() -> bool:
    """
    Check if client is initialized and connected to an index.
    
    Returns:
        bool: True if connected, False otherwise
    """
    return _client.pc is not None and _client.current_index is not None