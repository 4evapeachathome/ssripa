# RAG (Retrieval-Augmented Generation) Application

This is a Python-based RAG system that implements document processing, embedding generation, and similarity search for enhanced text generation.

## Project Structure

```
rag-app/
├── src/
│   ├── document_processor.py  # Document loading and chunking
│   ├── embedding_manager.py   # Embedding generation and storage
│   └── rag_engine.py         # Main RAG implementation
├── config/
│   └── config.py             # Configuration settings
├── data/                     # Your input documents
├── embeddings/              # Stored embeddings and indices
├── tests/                   # Test files
└── docs/                    # Documentation
```

## Features

- Document Processing
  - Support for TXT and PDF files
  - Intelligent document chunking
  - Directory-based document loading

- Embedding Generation
  - HuggingFace Embeddings support
  - OpenAI Embeddings support
  - FAISS-based similarity search

- Storage
  - Local file-based storage
  - FAISS index persistence
  - Metadata management

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_key_here  # If using OpenAI embeddings
   ```

## Usage

Basic usage example:

```python
from src.rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine()

# Process documents
rag.process_documents("data/")

# Query the system
results = rag.query("Your query here")
for result in results:
    print(f"Rank {result['rank']}: {result['content']}")
```

## Development Phases

1. Setup and Data Preparation ✓
   - Project scaffolding
   - Dependencies setup
   - Document preprocessing

2. Embedding and Storage ✓
   - Multiple embedding options
   - Local storage implementation
   - FAISS integration

3. Retrieval and Augmentation ✓
   - Similarity search
   - Context retrieval

4. Generation and Evaluation (In Progress)
   - LLM integration
   - System evaluation

## License

MIT License
