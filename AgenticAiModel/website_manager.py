#!/usr/bin/env python3
"""
Website Manager for RAG System

Manages multiple websites in the RAG system:
- Add new websites to the knowledge base
- List available websites
- Remove websites
- Query specific websites
- Update website content

Usage:
    python website_manager.py --add --website https://example.com --name "Example Website"
    python website_manager.py --list
    python website_manager.py --query "What is machine learning?" --website "Example Website"
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime

# Add paths for imports
sys.path.append('.')

from website_rag_pipeline import WebsiteRAGPipeline

class WebsiteManager:
    def __init__(self, config_file: str = "websites_config.json"):
        """
        Initialize the Website Manager.
        
        Args:
            config_file: JSON file to store website configurations
        """
        self.config_file = config_file
        self.websites = self.load_websites()
    
    def load_websites(self) -> Dict:
        """Load website configurations from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading website config: {e}")
                return {}
        return {}
    
    def save_websites(self):
        """Save website configurations to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.websites, f, indent=2)
        except Exception as e:
            print(f"Error saving website config: {e}")
    
    def add_website(self, website_url: str, website_name: str, chunk_size: int = 1000, 
                   overlap: int = 200, delay: float = 1.0, limit: Optional[int] = None) -> bool:
        """
        Add a new website to the RAG system.
        
        Args:
            website_url: URL of the website
            website_name: Name/identifier for the website
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            delay: Crawler delay
            limit: Maximum pages to crawl
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Adding website: {website_name} ({website_url})")
        
        # Create pipeline
        pipeline = WebsiteRAGPipeline(
            website_url=website_url,
            website_name=website_name,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        # Run pipeline
        success = pipeline.run_pipeline(delay=delay, limit=limit)
        
        if success:
            # Save website configuration
            self.websites[website_name] = {
                'url': website_url,
                'name': website_name,
                'chunk_size': chunk_size,
                'overlap': overlap,
                'added_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'status': 'active'
            }
            self.save_websites()
            
            print(f"✅ Successfully added website: {website_name}")
            return True
        else:
            print(f"❌ Failed to add website: {website_name}")
            return False
    
    def list_websites(self) -> None:
        """List all websites in the system."""
        if not self.websites:
            print("No websites configured.")
            return
        
        print("\n📚 Configured Websites:")
        print("-" * 80)
        
        for name, config in self.websites.items():
            status = config.get('status', 'unknown')
            added_at = config.get('added_at', 'unknown')
            last_updated = config.get('last_updated', 'unknown')
            
            print(f"🌐 {name}")
            print(f"   URL: {config['url']}")
            print(f"   Status: {status}")
            print(f"   Added: {added_at}")
            print(f"   Last Updated: {last_updated}")
            print(f"   Chunk Size: {config.get('chunk_size', 'N/A')}")
            print(f"   Overlap: {config.get('overlap', 'N/A')}")
            print()
    
    def remove_website(self, website_name: str) -> bool:
        """
        Remove a website from the system.
        
        Args:
            website_name: Name of the website to remove
            
        Returns:
            True if successful, False otherwise
        """
        if website_name not in self.websites:
            print(f"❌ Website '{website_name}' not found")
            return False
        
        try:
            # TODO: Implement Pinecone deletion for website-specific vectors
            # For now, just remove from config
            del self.websites[website_name]
            self.save_websites()
            
            print(f"✅ Removed website: {website_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error removing website: {e}")
            return False
    
    def update_website(self, website_name: str, delay: float = 1.0, limit: Optional[int] = None) -> bool:
        """
        Update website content by re-crawling.
        
        Args:
            website_name: Name of the website to update
            delay: Crawler delay
            limit: Maximum pages to crawl
            
        Returns:
            True if successful, False otherwise
        """
        if website_name not in self.websites:
            print(f"❌ Website '{website_name}' not found")
            return False
        
        config = self.websites[website_name]
        print(f"Updating website: {website_name}")
        
        # Create pipeline
        pipeline = WebsiteRAGPipeline(
            website_url=config['url'],
            website_name=website_name,
            chunk_size=config.get('chunk_size', 1000),
            overlap=config.get('overlap', 200)
        )
        
        # Run pipeline
        success = pipeline.run_pipeline(delay=delay, limit=limit)
        
        if success:
            # Update configuration
            config['last_updated'] = datetime.now().isoformat()
            self.save_websites()
            
            print(f"✅ Successfully updated website: {website_name}")
            return True
        else:
            print(f"❌ Failed to update website: {website_name}")
            return False
    
    def query_website(self, query: str, website_name: str, top_k: int = 5) -> List[Dict]:
        """
        Query a specific website.
        
        Args:
            query: User query
            website_name: Name of the website to query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if website_name not in self.websites:
            print(f"❌ Website '{website_name}' not found")
            return []
        
        config = self.websites[website_name]
        print(f"Querying website: {website_name}")
        
        # Create pipeline for querying
        pipeline = WebsiteRAGPipeline(
            website_url=config['url'],
            website_name=website_name
        )
        
        # Initialize Pinecone
        if not pipeline.initialize_pinecone():
            print("❌ Failed to initialize Pinecone")
            return []
        
        # Query website
        results = pipeline.query_website_specific(query, top_k=top_k)
        
        if results:
            print(f"✅ Found {len(results)} relevant documents")
            return results
        else:
            print("❌ No relevant documents found")
            return []


def main():
    """Main function to run the website manager from command line."""
    parser = argparse.ArgumentParser(description='Website Manager for RAG System')
    
    # Command groups
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--add', action='store_true', help='Add a new website')
    group.add_argument('--list', action='store_true', help='List all websites')
    group.add_argument('--remove', help='Remove a website by name')
    group.add_argument('--update', help='Update a website by name')
    group.add_argument('--query', help='Query a specific website')
    
    # Arguments for adding websites
    parser.add_argument('--website', help='Website URL (required for --add)')
    parser.add_argument('--name', help='Website name (required for --add)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size (default: 1000)')
    parser.add_argument('--overlap', type=int, default=200, help='Chunk overlap (default: 200)')
    parser.add_argument('--delay', type=float, default=1.0, help='Crawler delay (default: 1.0)')
    parser.add_argument('--limit', type=int, help='Maximum pages to crawl')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results for queries (default: 5)')
    
    args = parser.parse_args()
    
    # Create website manager
    manager = WebsiteManager()
    
    # Handle commands
    if args.add:
        if not args.website or not args.name:
            print("❌ --website and --name are required for --add")
            return
        
        manager.add_website(
            website_url=args.website,
            website_name=args.name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            delay=args.delay,
            limit=args.limit
        )
    
    elif args.list:
        manager.list_websites()
    
    elif args.remove:
        manager.remove_website(args.remove)
    
    elif args.update:
        manager.update_website(args.update, delay=args.delay, limit=args.limit)
    
    elif args.query:
        if not args.name:
            print("❌ --name is required for --query")
            return
        
        results = manager.query_website(args.query, args.name, top_k=args.top_k)
        
        if results:
            print(f"\n🔍 Query Results for '{args.name}':")
            print("-" * 80)
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. Score: {doc['score']:.3f}")
                print(f"   Source: {doc['source_url']}")
                print(f"   Text: {doc['text'][:200]}...")


if __name__ == "__main__":
    main() 