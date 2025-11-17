"""
Dense retriever using sentence transformers.
"""

import json
import numpy as np
from typing import Dict, List, Any
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers."""
    
    def __init__(self, collection_name: str, model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str = None, batch_size: int = 32):
        """
        Initialize dense retriever.
        
        Args:
            collection_name: Collection name
            model_name: HuggingFace model name for embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        super().__init__(collection_name)
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Model loaded on {self.device}")
        
        self.doc_ids = []
        self.doc_embeddings = None
        self.documents = {}
    
    def index(self, corpus: Dict[str, Dict[str, str]]):
        """
        Index corpus by encoding documents into embeddings.
        
        Args:
            corpus: Dictionary mapping document_id to {"text": ..., "title": ...}
        """
        print(f"Indexing {len(corpus)} documents with dense retriever...")
        
        self.doc_ids = []
        self.documents = {}
        texts = []
        
        # Prepare texts (combine title and text)
        for doc_id, doc in corpus.items():
            full_text = ""
            if doc.get("title"):
                full_text += doc["title"] + " "
            full_text += doc.get("text", "")
            
            self.doc_ids.append(doc_id)
            self.documents[doc_id] = doc
            texts.append(full_text)
        
        # Encode documents in batches
        print("Encoding documents...")
        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        self.indexed = True
        print(f"Indexed {len(self.doc_ids)} documents.")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query using cosine similarity.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.indexed:
            raise ValueError("Corpus must be indexed before retrieval. Call index() first.")
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute cosine similarity (dot product since embeddings are normalized)
        scores = np.dot(self.doc_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            results.append({
                "document_id": doc_id,
                "score": float(scores[idx]),
                "text": doc.get("text", ""),
                "title": doc.get("title", "")
            })
        
        return results

