"""
Hybrid retriever combining lexical (BM25) and dense retrievers.
"""

from typing import Dict, List, Any
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_retriever import BaseRetriever
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 and dense retrieval."""
    
    def __init__(self, collection_name: str, 
                 bm25_k1: float = 1.5, bm25_b: float = 0.75,
                 dense_model: str = "BAAI/bge-base-en-v1.5",
                 alpha: float = 0.5, fusion_method: str = "rrf"):
        """
        Initialize hybrid retriever.
        
        Args:
            collection_name: Collection name
            bm25_k1: BM25 parameter k1
            bm25_b: BM25 parameter b
            dense_model: Dense model name
            alpha: Weight for dense retrieval (1-alpha for BM25). Only used if fusion_method='weighted'
            fusion_method: Fusion method - 'rrf' (Reciprocal Rank Fusion) or 'weighted'
        """
        super().__init__(collection_name)
        self.alpha = alpha
        self.fusion_method = fusion_method
        
        # Initialize component retrievers
        self.bm25_retriever = BM25Retriever(collection_name, k1=bm25_k1, b=bm25_b)
        self.dense_retriever = DenseRetriever(collection_name, model_name=dense_model)
    
    def index(self, corpus: Dict[str, Dict[str, str]]):
        """
        Index corpus with both retrievers.
        
        Args:
            corpus: Dictionary mapping document_id to {"text": ..., "title": ...}
        """
        print("Indexing with hybrid retriever (BM25 + Dense)...")
        self.bm25_retriever.index(corpus)
        self.dense_retriever.index(corpus)
        self.indexed = True
    
    def _reciprocal_rank_fusion(self, bm25_results: List[Dict], dense_results: List[Dict], 
                               top_k: int, k: int = 60) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            bm25_results: BM25 retrieval results
            dense_results: Dense retrieval results
            top_k: Number of results to return
            k: RRF parameter (typically 60)
            
        Returns:
            Fused results
        """
        # Create score dictionaries
        bm25_scores = {doc["document_id"]: 1.0 / (k + rank + 1) 
                      for rank, doc in enumerate(bm25_results)}
        dense_scores = {doc["document_id"]: 1.0 / (k + rank + 1) 
                       for rank, doc in enumerate(dense_results)}
        
        # Combine scores
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        fused_scores = {}
        
        for doc_id in all_doc_ids:
            fused_scores[doc_id] = bm25_scores.get(doc_id, 0) + dense_scores.get(doc_id, 0)
        
        # Get document info
        doc_info = {}
        for doc in bm25_results + dense_results:
            if doc["document_id"] not in doc_info:
                doc_info[doc["document_id"]] = doc
        
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_info:
                results.append({
                    "document_id": doc_id,
                    "score": score,
                    "text": doc_info[doc_id].get("text", ""),
                    "title": doc_info[doc_id].get("title", "")
                })
        
        return results
    
    def _weighted_fusion(self, bm25_results: List[Dict], dense_results: List[Dict], 
                        top_k: int) -> List[Dict[str, Any]]:
        """
        Combine results using weighted score fusion.
        
        Args:
            bm25_results: BM25 retrieval results
            dense_results: Dense retrieval results
            top_k: Number of results to return
            
        Returns:
            Fused results
        """
        # Normalize scores to [0, 1]
        if bm25_results:
            max_bm25 = max(doc["score"] for doc in bm25_results)
            min_bm25 = min(doc["score"] for doc in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
        else:
            bm25_range = 1
        
        if dense_results:
            max_dense = max(doc["score"] for doc in dense_results)
            min_dense = min(doc["score"] for doc in dense_results)
            dense_range = max_dense - min_dense if max_dense != min_dense else 1
        else:
            dense_range = 1
        
        # Create score dictionaries with normalized scores
        bm25_scores = {}
        for doc in bm25_results:
            normalized = (doc["score"] - min_bm25) / bm25_range if bm25_range > 0 else 0.5
            bm25_scores[doc["document_id"]] = normalized
        
        dense_scores = {}
        for doc in dense_results:
            normalized = (doc["score"] - min_dense) / dense_range if dense_range > 0 else 0.5
            dense_scores[doc["document_id"]] = normalized
        
        # Combine scores
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        fused_scores = {}
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0)
            dense_score = dense_scores.get(doc_id, 0)
            fused_scores[doc_id] = (1 - self.alpha) * bm25_score + self.alpha * dense_score
        
        # Get document info
        doc_info = {}
        for doc in bm25_results + dense_results:
            if doc["document_id"] not in doc_info:
                doc_info[doc["document_id"]] = doc
        
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in sorted_docs:
            if doc_id in doc_info:
                results.append({
                    "document_id": doc_id,
                    "score": score,
                    "text": doc_info[doc_id].get("text", ""),
                    "title": doc_info[doc_id].get("title", "")
                })
        
        return results
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using hybrid retrieval.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.indexed:
            raise ValueError("Corpus must be indexed before retrieval. Call index() first.")
        
        # Retrieve from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        
        # Fuse results
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(bm25_results, dense_results, top_k)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(bm25_results, dense_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

