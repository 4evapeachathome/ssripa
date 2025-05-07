from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    DirectoryLoader
)
import os

class DocumentProcessor:
    """Handles document loading, preprocessing, and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def load_documents(self, path: str) -> List[Dict[str, Any]]:
        """Load documents from a file or directory."""
        if os.path.isfile(path):
            return self._load_file(path)
        elif os.path.isdir(path):
            return self._load_directory(path)
        else:
            raise ValueError(f"Invalid path: {path}")
    
    def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a single file based on its extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        return loader.load()
    
    def _load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Load all supported documents from a directory."""
        loader = DirectoryLoader(
            dir_path,
            glob="**/*.*",
            loader_cls=TextLoader
        )
        return loader.load()
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process(self, path: str) -> List[str]:
        """Load and chunk documents from the given path."""
        documents = self.load_documents(path)
        return self.chunk_documents(documents)
