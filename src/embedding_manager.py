from typing import List, Dict, Any
import numpy as np
import json
import os
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings
)
import faiss

class EmbeddingManager:
    """Manages document embeddings generation and storage."""
    
    def __init__(self, embedding_model: str = "huggingface"):
        self.embedding_model = self._initialize_embeddings(embedding_model)
        self.index = None
        
    def _initialize_embeddings(self, model_type: str):
        """Initialize the embedding model."""
        if model_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        elif model_type == "openai":
            return OpenAIEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding model: {model_type}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the given texts."""
        embeddings = self.embedding_model.embed_documents(texts)
        return np.array(embeddings)
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build a FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], 
                       metadata: Dict[str, Any], save_dir: str):
        """Save embeddings and metadata to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embeddings
        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
        
        # Save texts
        with open(os.path.join(save_dir, "texts.json"), "w") as f:
            json.dump(texts, f)
        
        # Save metadata
        with open(os.path.join(save_dir, "index.json"), "w") as f:
            json.dump(metadata, f)
        
        # Save FAISS index if it exists
        if self.index is not None:
            faiss.write_index(self.index, 
                            os.path.join(save_dir, "faiss_index.bin"))
    
    def load_embeddings(self, load_dir: str):
        """Load embeddings and metadata from disk."""
        embeddings = np.load(os.path.join(load_dir, "embeddings.npy"))
        
        with open(os.path.join(load_dir, "texts.json"), "r") as f:
            texts = json.load(f)
            
        with open(os.path.join(load_dir, "index.json"), "r") as f:
            metadata = json.load(f)
            
        if os.path.exists(os.path.join(load_dir, "faiss_index.bin")):
            self.index = faiss.read_index(
                os.path.join(load_dir, "faiss_index.bin")
            )
            
        return embeddings, texts, metadata
