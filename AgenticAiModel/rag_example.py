#!/usr/bin/env python3
"""
RAG System Example Usage

This script demonstrates how to use the RAG system with Pinecone integration.
It shows how to query the knowledge base with website-specific filtering.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

# Add Pinecone directory to path
sys.path.append('../Pinecone')

def setup_pinecone():
    """Setup Pinecone connection."""
    try:
        from Pinecone_utils import initialize_pinecone, connect_to_index
        
        print("🔧 Setting up Pinecone...")
        
        # Initialize Pinecone
        if not initialize_pinecone():
            print("❌ Failed to initialize Pinecone")
            return False
        
        # Connect to index
        if not connect_to_index():
            print("❌ Failed to connect to index")
            return False
        
        print("✅ Pinecone setup successful!")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone setup failed: {e}")
        return False

def get_available_websites():
    """Get list of available websites from Pinecone."""
    try:
        from Pinecone_utils import query_vectors, generate_embedding
        
        print("🌐 Getting available websites...")
        
        # Query to get available websites
        sample_query = "website"
        sample_embedding = generate_embedding(sample_query)
        
        # Get a sample of documents to find available websites
        sample_results = query_vectors(
            vector=sample_embedding,
            top_k=50,
            include_metadata=True,
            include_values=False
        )
        
        available_websites = set()
        for match in sample_results.matches:
            if hasattr(match, 'metadata') and match.metadata:
                website = match.metadata.get('website_name', '')
                if website:
                    available_websites.add(website)
        
        return sorted(list(available_websites))
        
    except Exception as e:
        print(f"❌ Error getting available websites: {e}")
        return []

def query_website_knowledge(query, website_name):
    """Query the knowledge base for a specific website."""
    try:
        from Pinecone_utils import query_vectors, generate_embedding
        from langchain_groq import ChatGroq
        
        print(f"🔍 Querying knowledge base for website: {website_name}")
        print(f"Query: {query}")
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Prepare filter for website-specific query
        filter_dict = {
            'website_name': {'$eq': website_name}
        }
        
        # Query Pinecone
        results = query_vectors(
            vector=query_embedding,
            top_k=5,
            filter=filter_dict,
            include_metadata=True,
            include_values=False
        )
        
        matches = results.matches if hasattr(results, 'matches') else []
        
        if not matches:
            return f"I couldn't find relevant information for '{query}' in the knowledge base for website '{website_name}'."
        
        # Extract retrieved documents
        retrieved_docs = []
        chunk_info = []
        
        for match in matches:
            if hasattr(match, 'metadata') and match.metadata:
                text = match.metadata.get('text', '')
                source_url = match.metadata.get('source_url', '')
                chunk_index = match.metadata.get('chunk_index', 'Unknown')
                total_chunks = match.metadata.get('total_chunks', 'Unknown')
                
                retrieved_docs.append(text)
                chunk_info.append({
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks,
                    'source_url': source_url,
                    'score': match.score
                })
                
                print(f"  📄 Chunk {chunk_index}/{total_chunks} (Score: {match.score:.4f})")
                print(f"     Source: {source_url}")
        
        # Build context from retrieved documents
        context = "\n\n".join(retrieved_docs)
        
        # Initialize LLM for answer generation
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "Error: GROQ_API_KEY not found in environment variables."
        
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        
        # Create a prompt for the LLM
        rag_prompt = f"""Based on the following retrieved information from website '{website_name}', please answer the user's question. 
        If the information is not sufficient to answer the question completely, acknowledge what you can answer and what additional information might be needed.

        Retrieved Information from {website_name}:
        {context}

        User Question: {query}

        Answer:"""
        
        # Generate response using the LLM
        llm_response = llm.invoke(rag_prompt)
        
        return llm_response.content
        
    except Exception as e:
        return f"Error in RAG query: {str(e)}"

def interactive_rag():
    """Interactive RAG system."""
    print("🤖 Interactive RAG System")
    print("=" * 50)
    
    # Setup Pinecone
    if not setup_pinecone():
        print("❌ Failed to setup Pinecone. Exiting.")
        return
    
    # Get available websites
    websites = get_available_websites()
    
    if not websites:
        print("❌ No websites found in the knowledge base.")
        return
    
    print(f"\n📋 Available websites: {', '.join(websites)}")
    
    while True:
        print("\n" + "=" * 50)
        print("Options:")
        print("1. Query a specific website")
        print("2. List available websites")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Query specific website
            print(f"\nAvailable websites: {', '.join(websites)}")
            website = input("Enter website name: ").strip()
            
            if website not in websites:
                print(f"❌ Website '{website}' not found in knowledge base.")
                continue
            
            query = input("Enter your question: ").strip()
            
            if not query:
                print("❌ Please enter a question.")
                continue
            
            print("\n🔍 Searching...")
            answer = query_website_knowledge(query, website)
            
            print(f"\n📝 Answer:")
            print("-" * 30)
            print(answer)
            
        elif choice == "2":
            # List websites
            print(f"\n📋 Available websites: {', '.join(websites)}")
            
        elif choice == "3":
            # Exit
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

def example_queries():
    """Run example queries."""
    print("📚 Example Queries")
    print("=" * 50)
    
    # Setup Pinecone
    if not setup_pinecone():
        print("❌ Failed to setup Pinecone. Exiting.")
        return
    
    # Get available websites
    websites = get_available_websites()
    
    if not websites:
        print("❌ No websites found in the knowledge base.")
        return
    
    print(f"Available websites: {', '.join(websites)}")
    
    # Example queries
    example_queries = [
        "What is machine learning?",
        "How does JavaScript work?",
        "What are the main features of the website?",
        "Tell me about the courses offered",
        "What is the contact information?"
    ]
    
    # Use the first available website for examples
    website = websites[0]
    print(f"\nUsing website: {website}")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n📝 Example {i}: {query}")
        print("-" * 40)
        
        answer = query_website_knowledge(query, website)
        print(answer)
        
        if i < len(example_queries):
            input("\nPress Enter to continue to next example...")

def main():
    """Main function."""
    print("🚀 RAG System Example")
    print("=" * 50)
    
    print("Choose an option:")
    print("1. Interactive RAG system")
    print("2. Run example queries")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        interactive_rag()
    elif choice == "2":
        example_queries()
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main() 