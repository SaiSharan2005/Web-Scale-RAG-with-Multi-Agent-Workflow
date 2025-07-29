# Website RAG System

A complete Retrieval-Augmented Generation (RAG) system that crawls websites, chunks content, stores it in Pinecone, and provides website-specific question answering.

## 🚀 Features

- **Multi-Website Support**: Manage multiple websites in a single RAG system
- **Website-Specific Filtering**: Query specific websites using `website: WebsiteName` syntax
- **Automatic Crawling**: Crawl entire websites with configurable limits and delays
- **Smart Chunking**: Sliding window chunking with overlap for better context
- **Pinecone Integration**: Store and retrieve vectors with metadata filtering
- **Multi-Agent Workflow**: Integrates with your existing multi-agent architecture

## 📋 Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install langchain langchain_groq langchain_community langgraph rizaio python-dotenv pinecone-client beautifulsoup4 requests tqdm
   ```

2. **Set up API Keys** in `config.env`:
   ```env
   GROQ_API_KEY=your_groq_api_key
   RIZA_API_KEY=your_riza_api_key
   TAVILY_API_KEY=your_tavily_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=website-rag-index
   ```

## 🏗️ System Architecture

```
User Query → Multi-Agent System → RAG Node → Pinecone (with website filter) → LLM Response
```

### Components:
1. **Website RAG Pipeline** (`website_rag_pipeline.py`): Crawls, chunks, and stores website content
2. **Website Manager** (`website_manager.py`): Manages multiple websites
3. **Enhanced RAG Node** (`2.py`): Retrieves website-specific content
4. **Pinecone Utils** (`../Pinecone/Pinecone_utils.py`): Vector database operations

## 🎯 Quick Start

### 1. Add a Website to the RAG System

```bash
# Add a website
python website_manager.py --add --website https://example.com --name "Example Website"

# Add with custom settings
python website_manager.py --add \
  --website https://example.com \
  --name "Example Website" \
  --chunk-size 1000 \
  --overlap 200 \
  --delay 1.0 \
  --limit 50
```

### 2. List Available Websites

```bash
python website_manager.py --list
```

### 3. Query a Specific Website

```bash
# Query a specific website
python website_manager.py --query "What is machine learning?" --name "Example Website"
```

### 4. Use in Multi-Agent System

```python
# Run the multi-agent system with website-specific queries
python 2.py
```

## 🔧 Usage Examples

### Adding Multiple Websites

```bash
# Add a tech blog
python website_manager.py --add --website https://techblog.example.com --name "Tech Blog"

# Add a company website
python website_manager.py --add --website https://company.example.com --name "Company Website"

# Add documentation site
python website_manager.py --add --website https://docs.example.com --name "Documentation"
```

### Website-Specific Queries

In the multi-agent system, use the `website:` prefix to query specific websites:

```
User: "What is machine learning? website: Tech Blog"
User: "How does the company handle support? website: Company Website"
User: "What are the API endpoints? website: Documentation"
```

### Updating Website Content

```bash
# Update a website (re-crawl)
python website_manager.py --update "Example Website" --limit 100
```

### Removing Websites

```bash
# Remove a website
python website_manager.py --remove "Example Website"
```

## 🏛️ Multi-Agent Integration

The RAG system integrates seamlessly with your multi-agent architecture:

### Agent Routing:
- **Enhancer**: Clarifies vague queries
- **Researcher**: Gets real-time information from the web
- **RAG**: Retrieves information from stored website knowledge

### Query Flow:
1. User asks: `"What is machine learning? website: Tech Blog"`
2. Supervisor routes to RAG agent
3. RAG agent filters Pinecone for "Tech Blog" content
4. Retrieves relevant chunks and generates answer
5. Validator ensures quality and completeness

## 📊 Configuration Options

### Crawler Settings:
- `--delay`: Time between requests (default: 1.0s)
- `--limit`: Maximum pages to crawl (default: unlimited)
- `--chunk-size`: Size of text chunks (default: 1000)
- `--overlap`: Overlap between chunks (default: 200)

### Pinecone Settings:
- `PINECONE_INDEX_NAME`: Index name (default: website-rag-index)
- `PINECONE_TOP_K`: Number of results (default: 5)
- `PINECONE_BATCH_SIZE`: Batch size for upserts (default: 100)

## 🔍 Query Examples

### General Queries:
```
"What's the weather in Hyderabad today?"
"What is the difference between the stock price of Apple in 2023 and 2021?"
"Research the impact of climate change on agriculture in Southeast Asia"
```

### Website-Specific Queries:
```
"What is machine learning? website: Tech Blog"
"How does the company handle customer support? website: Company Website"
"What are the API endpoints? website: Documentation"
"Tell me about the product features website: Product Site"
```

## 🛠️ Advanced Usage

### Custom Chunking Strategy

```python
from website_rag_pipeline import WebsiteRAGPipeline

pipeline = WebsiteRAGPipeline(
    website_url="https://example.com",
    website_name="Example",
    chunk_size=1500,  # Larger chunks
    overlap=300       # More overlap
)
```

### Batch Processing

```python
from website_manager import WebsiteManager

manager = WebsiteManager()

# Add multiple websites
websites = [
    ("https://blog1.example.com", "Blog 1"),
    ("https://blog2.example.com", "Blog 2"),
    ("https://docs.example.com", "Documentation")
]

for url, name in websites:
    manager.add_website(url, name, limit=100)
```

### Custom Filtering

```python
# Query with custom filters
results = pipeline.query_website_specific(
    query="machine learning",
    top_k=10,
    filter={
        'website_name': {'$eq': 'Tech Blog'},
        'chunk_size': {'$gte': 500}
    }
)
```

## 🚨 Troubleshooting

### Common Issues:

1. **Pinecone Connection Error**:
   - Check your `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`
   - Ensure the index exists and is accessible

2. **Crawler Issues**:
   - Check if the website allows crawling (robots.txt)
   - Increase delay if getting rate-limited
   - Check network connectivity

3. **No Results Found**:
   - Verify the website name in your query
   - Check if the website was successfully added
   - Try rephrasing your question

### Debug Mode:

```bash
# Enable verbose logging
python website_manager.py --add --website https://example.com --name "Example" --verbose
```

## 📈 Performance Tips

1. **Chunk Size**: Larger chunks (1000-1500) for detailed content, smaller (500-800) for FAQs
2. **Overlap**: 20-30% overlap for better context preservation
3. **Batch Size**: Use larger batch sizes (100-200) for faster Pinecone operations
4. **Crawler Delay**: 1-2 seconds to be respectful to servers

## 🔄 Maintenance

### Regular Updates:
```bash
# Update all websites weekly
python website_manager.py --update "Tech Blog"
python website_manager.py --update "Company Website"
```

### Monitoring:
- Check Pinecone index statistics
- Monitor crawl success rates
- Review query performance

## 📝 File Structure

```
AgenticAiModel/
├── 2.py                          # Main multi-agent system with RAG
├── website_rag_pipeline.py       # Complete RAG pipeline
├── website_manager.py            # Website management utility
├── config.env                    # API keys and configuration
├── websites_config.json          # Website configurations
└── README_WEBSITE_RAG.md         # This file

Crawler/
├── crawler.py                    # Web crawler
└── crawler.env                   # Crawler configuration

Chunking/
└── chunking.py                   # Text chunking system

Pinecone/
├── Pinecone_utils.py             # Pinecone utilities
└── pinecone.env                  # Pinecone configuration
```

## 🎉 Success!

You now have a complete Website RAG system that can:
- Crawl any website
- Store content in Pinecone with website-specific metadata
- Query specific websites using natural language
- Integrate with your multi-agent architecture

Start by adding a few websites and testing the system with website-specific queries! 