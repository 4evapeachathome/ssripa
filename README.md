# SSRIPA RAG System

A modern Retrieval-Augmented Generation (RAG) system built with FastAPI and Azure Functions, leveraging state-of-the-art language models and vector stores for efficient document processing and retrieval.

## Tech Stack

### Core Infrastructure
- FastAPI
- Azure Functions
- Python 3.9

### RAG Components
- LangChain 0.1.0
- ChromaDB 0.4.22
- FAISS-CPU 1.7.4
- OpenAI 1.0.0

### Embedding Models
- HuggingFace Sentence Transformers 2.2.2
- OpenAI Ada

### Document Processing
- PyPDF 4.0.1
- python-magic 0.4.27
- Tiktoken 0.5.2

### Data & Utilities
- NumPy 1.24.3
- Pandas 2.1.0
- python-dotenv 1.0.0
- Pydantic 2.4.2

### Testing & Development
- Pytest 7.4.2
- Uvicorn 0.24.0

### Deployment
- GitHub Actions
- Azure Functions

## Features

- **Advanced Document Processing**
  - PDF and text document support
  - Intelligent text chunking with overlap
  - Automatic metadata extraction
  - Token-aware document splitting

- **Flexible Embedding Generation**
  - Dual embedding model support (HuggingFace and OpenAI)
  - Efficient batch processing
  - Vector store persistence

- **Intelligent Retrieval**
  - Hybrid search capabilities
  - Context-aware document retrieval
  - Semantic similarity ranking
  - FAISS-powered vector search

- **API Integration**
  - RESTful API endpoints
  - Azure Functions serverless deployment
  - Async request handling
  - Structured response formats

## Setup

1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   AZURE_FUNCTION_KEY=your_azure_key
   ```

## Usage

### API Endpoints

```python
# Document Processing
POST /process-documents
- Uploads and processes documents
- Returns processing status and document IDs

# Query Generation
POST /generate-answer
- Accepts user query and optional parameters
- Returns RAG-enhanced response with sources

# Vector Store Management
GET /vector-store-info
- Retrieves vector store statistics
- Shows embedding distribution
```

### Example Query

```python
import requests

response = requests.post(
    "http://localhost:8000/generate-answer",
    json={
        "query": "What are the key features?",
        "max_tokens": 500,
        "temperature": 0.7
    }
)

print(response.json())
```

## Development Status

✅ Core RAG Implementation
✅ Document Processing Pipeline
✅ Embedding Generation System
✅ Vector Store Integration
✅ API Development
✅ Azure Functions Deployment
🔄 Continuous Improvements & Optimizations

## License

MIT License
