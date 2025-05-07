from typing import List, Dict, Any
import numpy as np
from .document_processor import DocumentProcessor
from .embedding_manager import EmbeddingManager

class RAGEngine:
    """Main RAG engine that coordinates document processing, embedding, and retrieval."""
    
    def __init__(self, embedding_model: str = "huggingface"):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.documents = []
        self.embeddings = None
        
    def process_documents(self, path: str):
        """Process documents from the given path."""
        self.documents = self.document_processor.process(path)
        
        # Generate and store embeddings
        self.embeddings = self.embedding_manager.generate_embeddings(self.documents)
        self.embedding_manager.build_faiss_index(self.embeddings)
        
        # Save everything
        metadata = {
            "num_documents": len(self.documents),
            "embedding_dim": self.embeddings.shape[1],
            "source_path": path
        }
        self.embedding_manager.save_embeddings(
            self.embeddings,
            self.documents,
            metadata,
            "embeddings"
        )
        
    def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Query the RAG system."""
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query_text])
        
        # Perform similarity search
        if self.embedding_manager.index is None:
            raise ValueError("No index available. Process documents first.")
            
        D, I = self.embedding_manager.index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            results.append({
                "content": self.documents[idx],
                "score": float(distance),
                "rank": i + 1
            })
            
        return results
    
    def load_existing_index(self, load_dir: str = "embeddings"):
        """Load existing embeddings and index."""
        self.embeddings, self.documents, metadata = \
            self.embedding_manager.load_embeddings(load_dir)
        return metadata
