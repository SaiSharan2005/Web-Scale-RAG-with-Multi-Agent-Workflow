# AgenticAI Model - Multi-Agent RAG System

A comprehensive multi-agent system that combines web crawling, RAG (Retrieval-Augmented Generation), and intelligent workflow management to provide accurate, contextual answers from website knowledge bases.

## 🏗️ System Architecture

```
AgenticAiModel/
├── multi_agent_class.py      # Core multi-agent workflow system
├── website_rag_pipeline.py   # Website crawling and RAG pipeline
├── website_manager.py        # Website management interface
├── websites_config.json      # Website configurations
├── config.env               # API keys and configuration
└── README_WEBSITE_RAG.md    # Detailed RAG documentation
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install langchain langchain-groq langchain-community langgraph pydantic python-dotenv

# Set up API keys in config.env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
RIZA_API_KEY=your_riza_api_key
```

### 2. Initialize Pinecone
```bash
# Set up Pinecone in ../Pinecone/pinecone.env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-model-agentic-ai
PINECONE_ENVIRONMENT=your_environment
```

### 3. Add a Website
```bash
python website_manager.py --add --website https://example.com --name "Example Website"
```

### 4. Run Multi-Agent System
```bash
python multi_agent_class.py
```

## 📁 File Documentation

### 1. `multi_agent_class.py` - Core Multi-Agent System

**Purpose**: Main multi-agent workflow system with intelligent routing and validation.

**Key Features**:
- **6 Specialized Agents**: Supervisor, Enhancer, Researcher, Web Search, RAG, Validator
- **Smart Routing**: Intelligent task distribution based on query type
- **Quality Control**: Enhanced validation with specific rejection criteria
- **Loop Prevention**: Node usage counters prevent infinite cycles
- **Multiple Search Sources**: Tavily, DuckDuckGo, and Pinecone RAG

**Usage**:
```python
from multi_agent_class import MultiAgentWorkflow

# Initialize with website and debug mode
workflow = MultiAgentWorkflow(website_name="Piyush", debug_mode=True)

# Run a query
outputs = workflow.run_query("What courses does Piyush offer?")
```

**Agent Roles**:
- **Supervisor**: Routes queries to appropriate agents
- **Enhancer**: Clarifies vague or incomplete queries
- **Researcher**: Uses Tavily for general web research
- **Web Search**: Uses DuckDuckGo for real-time information
- **RAG**: Retrieves from Pinecone knowledge base
- **Validator**: Ensures answer quality and completeness

### 2. `website_rag_pipeline.py` - Website Processing Pipeline

**Purpose**: Complete pipeline for crawling, chunking, and storing website content in Pinecone.

**Pipeline Steps**:
1. **Website Crawling**: Uses DomainCrawler to extract content
2. **Text Chunking**: Splits content into manageable chunks
3. **Vector Generation**: Creates embeddings for each chunk
4. **Pinecone Storage**: Stores vectors with website-specific metadata

**Usage**:
```python
from website_rag_pipeline import WebsiteRAGPipeline

pipeline = WebsiteRAGPipeline(
    website_url="https://example.com",
    website_name="Example",
    chunk_size=1000,
    overlap=200
)

success = pipeline.run_pipeline(delay=1.0, limit=50)
```

**Key Methods**:
- `crawl_website()`: Extracts content from website
- `chunk_content()`: Splits content into chunks
- `store_in_pinecone()`: Stores vectors in Pinecone
- `query_website_specific()`: Queries website-specific knowledge

### 3. `website_manager.py` - Website Management Interface

**Purpose**: Command-line interface for managing websites in the RAG system.

**Features**:
- Add new websites to knowledge base
- List available websites
- Remove websites
- Update website content
- Query specific websites

**Usage**:
```bash
# Add a new website
python website_manager.py --add --website https://example.com --name "Example"

# List all websites
python website_manager.py --list

# Query a specific website
python website_manager.py --query "What is machine learning?" --website "Example"

# Update website content
python website_manager.py --update --website "Example"

# Remove a website
python website_manager.py --remove --website "Example"
```

**Key Methods**:
- `add_website()`: Add new website to system
- `list_websites()`: Display all configured websites
- `remove_website()`: Remove website from system
- `update_website()`: Refresh website content
- `query_website()`: Query specific website knowledge

### 4. `websites_config.json` - Website Configuration

**Purpose**: JSON configuration file storing website metadata and settings.

**Structure**:
```json
{
  "WebsiteName": {
    "url": "https://example.com",
    "name": "WebsiteName",
    "chunk_size": 1000,
    "overlap": 200,
    "added_at": "2025-01-01T00:00:00",
    "last_updated": "2025-01-01T00:00:00",
    "status": "active"
  }
}
```

**Fields**:
- `url`: Website URL
- `name`: Website identifier
- `chunk_size`: Text chunk size in characters
- `overlap`: Overlap between chunks
- `added_at`: Timestamp when added
- `last_updated`: Last update timestamp
- `status`: Website status (active/inactive)

### 5. `config.env` - Environment Configuration

**Purpose**: Environment variables for API keys and system configuration.

**Required Variables**:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
RIZA_API_KEY=your_riza_api_key
```

## 🔧 Configuration

### API Keys Setup

1. **Groq API**: For LLM processing
   - Sign up at https://console.groq.com/
   - Get API key from dashboard

2. **Tavily API**: For web search
   - Sign up at https://tavily.com/
   - Get API key from dashboard

3. **Pinecone API**: For vector storage
   - Sign up at https://www.pinecone.io/
   - Get API key and environment

### Pinecone Setup

Create `../Pinecone/pinecone.env`:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-model-agentic-ai
PINECONE_ENVIRONMENT=your_environment
```

## 🎯 Use Cases

### 1. Website Knowledge Base
```bash
# Add a company website
python website_manager.py --add --website https://company.com --name "Company"

# Query company information
python multi_agent_class.py
# Enter website name: Company
# Enter query: What services does the company offer?
```

### 2. Educational Content
```bash
# Add educational website
python website_manager.py --add --website https://course.com --name "Courses"

# Query course information
python multi_agent_class.py
# Enter website name: Courses
# Enter query: What are the prerequisites for the Python course?
```

### 3. Documentation Search
```bash
# Add documentation site
python website_manager.py --add --website https://docs.example.com --name "Docs"

# Query documentation
python multi_agent_class.py
# Enter website name: Docs
# Enter query: How do I implement authentication?
```

## 🔄 Workflow Process

### 1. Website Addition
```
User Input → Website Manager → RAG Pipeline → Pinecone Storage
```

### 2. Query Processing
```
User Query → Supervisor → Agent Selection → Processing → Validator → Final Answer
```

### 3. Multi-Agent Routing
```
Query → Enhancer (if unclear) → Supervisor → Researcher/Web Search/RAG → Validator
```

## 🛠️ Advanced Features

### Debug Mode
Enable detailed workflow tracking:
```python
workflow = MultiAgentWorkflow(website_name="Example", debug_mode=True)
```

### Node Usage Limits
Prevents infinite loops by limiting each node to 2 uses per query.

### Smart Validation
Rejects answers that:
- Say "I couldn't find information"
- Ask for more details
- Mention wrong topics
- Are vague or incomplete

### Multiple Search Sources
- **Tavily**: General web research
- **DuckDuckGo**: Real-time information
- **Pinecone RAG**: Stored knowledge base

## 📊 Performance Optimization

### Chunking Strategy
- Default chunk size: 1000 characters
- Overlap: 200 characters
- Configurable per website

### Crawling Limits
- Default delay: 1 second between requests
- Configurable page limits
- Respects robots.txt

### Vector Storage
- Pinecone for scalable vector storage
- Website-specific filtering
- Metadata for efficient retrieval

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Check config.env file
   cat config.env
   # Ensure all required keys are present
   ```

2. **Pinecone Connection Issues**
   ```bash
   # Check pinecone.env in ../Pinecone/
   cat ../Pinecone/pinecone.env
   # Verify API key and environment
   ```

3. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

4. **Website Crawling Issues**
   ```bash
   # Check website accessibility
   curl -I https://example.com
   # Verify robots.txt compliance
   ```

### Debug Commands

```bash
# Enable debug mode
python multi_agent_class.py
# Enter 'y' for debug mode

# Check website status
python website_manager.py --list

# Test specific website
python website_manager.py --query "test" --website "WebsiteName"
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Support for non-English websites
- **Real-time Updates**: Automatic website content updates
- **Advanced Analytics**: Query performance metrics
- **Custom Embeddings**: Support for different embedding models
- **API Endpoints**: REST API for integration

### Scalability Improvements
- **Distributed Processing**: Multi-node processing
- **Caching Layer**: Redis for query caching
- **Load Balancing**: Multiple Pinecone indexes
- **Async Processing**: Non-blocking operations

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**AgenticAI Model** - Intelligent Multi-Agent RAG System for Website Knowledge Management 