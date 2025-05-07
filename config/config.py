from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the RAG application."""
    
    # Embedding settings
    EMBEDDING_MODEL = "huggingface"  # or "openai"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Document processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    TOP_K_RESULTS = 3
    
    # Storage settings
    EMBEDDINGS_DIR = "embeddings"
    DATA_DIR = "data"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items() 
            if not key.startswith("_")
        }
