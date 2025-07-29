# AgenticAI Model - Quick Reference Guide

## 🚀 Quick Commands

### Setup
```bash
# Install dependencies
pip install langchain langchain-groq langchain-community langgraph pydantic python-dotenv

# Set up API keys
cp config.env.example config.env
# Edit config.env with your API keys
```

### Website Management
```bash
# Add website
python website_manager.py --add --website https://example.com --name "Example"

# List websites
python website_manager.py --list

# Query website
python website_manager.py --query "What is AI?" --website "Example"

# Update website
python website_manager.py --update --website "Example"

# Remove website
python website_manager.py --remove --website "Example"
```

### Multi-Agent System
```bash
# Run interactive mode
python multi_agent_class.py

# Programmatic usage
python -c "
from multi_agent_class import MultiAgentWorkflow
workflow = MultiAgentWorkflow(website_name='Example', debug_mode=True)
outputs = workflow.run_query('What courses are available?')
"
```

## 📁 File Structure

```
AgenticAiModel/
├── multi_agent_class.py      # Main multi-agent system
├── website_rag_pipeline.py   # Website processing pipeline
├── website_manager.py        # Website management CLI
├── websites_config.json      # Website configurations
├── config.env               # API keys
├── README.md                # Main documentation
├── DOCUMENTATION.md         # Technical documentation
├── QUICK_REFERENCE.md       # This file
└── README_WEBSITE_RAG.md    # RAG system docs
```

## 🔧 Configuration

### Required API Keys
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
RIZA_API_KEY=your_riza_api_key
```

### Pinecone Setup
```env
# ../Pinecone/pinecone.env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-model-agentic-ai
PINECONE_ENVIRONMENT=your_environment
```

## 🎯 Use Cases

### 1. Company Knowledge Base
```bash
# Add company website
python website_manager.py --add --website https://company.com --name "Company"

# Query company info
python multi_agent_class.py
# Website: Company
# Query: What services does the company offer?
```

### 2. Educational Content
```bash
# Add course website
python website_manager.py --add --website https://courses.com --name "Courses"

# Query course info
python multi_agent_class.py
# Website: Courses
# Query: What are the prerequisites for Python course?
```

### 3. Documentation Search
```bash
# Add docs website
python website_manager.py --add --website https://docs.example.com --name "Docs"

# Query documentation
python multi_agent_class.py
# Website: Docs
# Query: How do I implement authentication?
```

## 🔄 Workflow Overview

### Multi-Agent Flow
```
User Query → Supervisor → Agent Selection → Processing → Validator → Final Answer
```

### Agent Types
- **Supervisor**: Routes queries to appropriate agents
- **Enhancer**: Clarifies vague queries
- **Researcher**: Uses Tavily for web research
- **Web Search**: Uses DuckDuckGo for real-time info
- **RAG**: Retrieves from Pinecone knowledge base
- **Validator**: Ensures answer quality

### Routing Logic
- **Enhancer**: Unclear/vague queries
- **Researcher**: Current, real-time information
- **Web Search**: Breaking news, real-time events
- **RAG**: Stored knowledge, courses, website-specific info

## 🛠️ Debug Mode

### Enable Debug
```python
workflow = MultiAgentWorkflow(website_name="Example", debug_mode=True)
```

### Debug Output
```
Current Node: Supervisor -> Goto: rag
Reason: Query about stored knowledge, routing to RAG
Node usage counters: {'researcher': 0, 'rag': 1, 'enhancer': 0, 'web_search': 0}
```

## 📊 Performance Tuning

### Chunking Parameters
```bash
# Custom chunk size and overlap
python website_manager.py --add --website https://example.com --name "Example" --chunk-size 1500 --overlap 300
```

### Crawling Limits
```bash
# Limit pages and set delay
python website_manager.py --add --website https://example.com --name "Example" --limit 50 --delay 2.0
```

### Query Parameters
```bash
# Custom number of results
python website_manager.py --query "What is AI?" --website "Example" --top-k 10
```

## 🚨 Common Issues

### API Key Errors
```bash
# Check config.env
cat config.env

# Verify Pinecone setup
cat ../Pinecone/pinecone.env
```

### Import Errors
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Website Crawling Issues
```bash
# Test website accessibility
curl -I https://example.com

# Check robots.txt
curl https://example.com/robots.txt
```

## 📈 Monitoring

### Node Usage Tracking
```python
# Check node counters
print(workflow.node_counters)
# Output: {'researcher': 1, 'rag': 2, 'enhancer': 0, 'web_search': 0}
```

### Validation Results
```python
# Debug validation decisions
# Output shows validator reasoning and routing decisions
```

## 🔮 Advanced Features

### Custom Embeddings
```python
# Modify embedding model in Pinecone_utils.py
# Change model in generate_embedding function
```

### Custom Chunking
```python
# Modify chunking strategy in website_rag_pipeline.py
# Adjust chunk_size and overlap parameters
```

### Custom Validation
```python
# Modify validation criteria in multi_agent_class.py
# Update system_prompt in _validator_node
```

## 📞 Support

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

### Log Files
- Check console output for detailed logs
- Debug mode provides step-by-step workflow tracking
- Error messages include troubleshooting hints

---

**Quick Reference Guide** - AgenticAI Model v1.0 