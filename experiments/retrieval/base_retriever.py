"""
Base retriever interface for Task A (Retrieval Only).
All retrievers should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json


class BaseRetriever(ABC):
    """Base class for all retrievers."""
    
    def __init__(self, collection_name: str):
        """
        Initialize the retriever.
        
        Args:
            collection_name: Name of the collection (e.g., "mt-rag-clapnq-elser-512-100-20240503")
        """
        self.collection_name = collection_name
        self.indexed = False
    
    @abstractmethod
    def index(self, corpus: Dict[str, Dict[str, str]]):
        """
        Index the corpus for retrieval.
        
        Args:
            corpus: Dictionary mapping document_id to {"text": ..., "title": ...}
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries with keys: document_id, score, text, title
        """
        pass
    
    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries (default implementation).
        
        Args:
            queries: List of query texts
            top_k: Number of documents to retrieve per query
            
        Returns:
            List of lists of retrieved documents
        """
        return [self.retrieve(query, top_k) for query in queries]
    
    def format_results(self, query_id: str, retrieved_docs: List[Dict[str, Any]], 
                      collection_name: str) -> Dict[str, Any]:
        """
        Format retrieval results for evaluation script.
        
        Args:
            query_id: Query ID (task_id)
            retrieved_docs: List of retrieved documents
            collection_name: Collection name
            
        Returns:
            Formatted result dictionary
        """
        return {
            "task_id": query_id,
            "Collection": collection_name,
            "contexts": [
                {
                    "document_id": doc["document_id"],
                    "score": doc["score"],
                    "text": doc.get("text", ""),
                    "title": doc.get("title", "")
                }
                for doc in retrieved_docs
            ]
        }

