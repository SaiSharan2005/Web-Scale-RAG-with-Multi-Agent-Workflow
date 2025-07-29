import os
from Pinecone_utils import initialize_pinecone, connect_to_index, generate_embedding, query_vectors

# Configuration
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'rag-model-agentic-ai')
WEBSITE_NAME = 'Elancode'
USER_QUERY = 'Your question about Elancode here'
TOP_K = 5


def main():
    # 1. Initialize and connect
    if not initialize_pinecone():
        raise SystemExit('Failed to initialize Pinecone client')

    if not connect_to_index(INDEX_NAME):
        raise SystemExit(f'Failed to connect to index: {INDEX_NAME}')

    # 2. Generate embedding for the user query
    embedding = generate_embedding(USER_QUERY)
    print(f"Query embedding generated (len={len(embedding)})")

    # 3. Run a filtered query by website_name
    filter_dict = {'website_name': {'$eq': WEBSITE_NAME}}
    response = query_vectors(
        vector=embedding,
        top_k=TOP_K,
        filter=filter_dict,
        include_metadata=True,
        include_values=False
    )

    # 4. Display the nearest chunks
    print(f"\nTop {TOP_K} chunks for '{USER_QUERY}' in site '{WEBSITE_NAME}':\n")
    if hasattr(response, 'matches') and response.matches:
        for match in response.matches:
            meta = match.metadata or {}
            text = meta.get('text', '<no text>')
            url = meta.get('source_url', '<no url>')
            score = getattr(match, 'score', None)
            print(f"- Score: {score:.4f}\n  URL: {url}\n  Chunk: {text}\n")
    else:
        print("No relevant chunks found.")


if __name__ == '__main__':
    main()
