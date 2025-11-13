"""
Re-ranker for improving retrieval results using cross-encoders.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Any
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


class Reranker:
    """Re-ranker using cross-encoder models."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 device: str = None, batch_size: int = 32):
        """
        Initialize re-ranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for re-ranking
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        print(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name, device=self.device)
        print(f"Re-ranker loaded on {self.device}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: int = None) -> List[Dict[str, Any]]:
        """
        Re-rank documents for a query.
        
        Args:
            query: Query text
            documents: List of documents with 'text' and 'document_id'
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            Re-ranked list of documents
        """
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.get("text", "")] for doc in documents]
        
        # Get scores in batches
        scores = []
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Re-ranking", leave=False):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch_pairs)
            scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)
        
        # Combine documents with scores
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_docs.append({
                "document_id": doc["document_id"],
                "score": float(score),
                "text": doc.get("text", ""),
                "title": doc.get("title", "")
            })
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k
        if top_k is not None:
            return scored_docs[:top_k]
        return scored_docs


class RetrievalReranker:
    """Wrapper that adds re-ranking to any retriever."""
    
    def __init__(self, base_retriever, reranker: Reranker = None, 
                 initial_top_k: int = 100, final_top_k: int = 10):
        """
        Initialize retrieval with re-ranking.
        
        Args:
            base_retriever: Base retriever instance
            reranker: Re-ranker instance (if None, creates default)
            initial_top_k: Number of documents to retrieve before re-ranking
            final_top_k: Number of documents to return after re-ranking
        """
        self.base_retriever = base_retriever
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        
        if reranker is None:
            self.reranker = Reranker()
        else:
            self.reranker = reranker
    
    def index(self, corpus):
        """Index corpus using base retriever."""
        self.base_retriever.index(corpus)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve and re-rank documents.
        
        Args:
            query: Query text
            top_k: Final number of documents (overrides final_top_k)
            
        Returns:
            Re-ranked list of documents
        """
        if top_k is None:
            top_k = self.final_top_k
        
        # Step 1: Initial retrieval (get more documents)
        initial_results = self.base_retriever.retrieve(query, top_k=self.initial_top_k)
        
        if not initial_results:
            return []
        
        # Step 2: Re-rank
        reranked = self.reranker.rerank(query, initial_results, top_k=top_k)
        
        return reranked

