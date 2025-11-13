"""
Multi-query retrieval for handling multi-turn conversations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Any
import numpy as np
from base_retriever import BaseRetriever
from query_processor import extract_last_turn, extract_all_user_turns


class MultiQueryRetriever:
    """Retriever that handles multi-turn queries by retrieving for each turn."""
    
    def __init__(self, base_retriever: BaseRetriever, fusion_method: str = "rrf", 
                 k: int = 60):
        """
        Initialize multi-query retriever.
        
        Args:
            base_retriever: Base retriever to use
            fusion_method: Fusion method ('rrf', 'max', 'mean', 'weighted')
            k: RRF parameter (if using RRF)
        """
        self.base_retriever = base_retriever
        self.fusion_method = fusion_method
        self.k = k
    
    def index(self, corpus: Dict[str, Dict[str, str]]):
        """Index corpus using base retriever."""
        self.base_retriever.index(corpus)
    
    def _extract_turns(self, query_text: str) -> List[str]:
        """Extract individual user turns from multi-turn query."""
        import re
        lines = query_text.split('\n')
        user_turns = []
        for line in lines:
            if line.strip().startswith('|user|:'):
                turn = re.sub(r'\|user\|:\s*', '', line).strip()
                if turn:
                    user_turns.append(turn)
        return user_turns if user_turns else [query_text.strip()]
    
    def _fuse_results_rrf(self, all_results: List[List[Dict]], top_k: int) -> List[Dict]:
        """Fuse results using Reciprocal Rank Fusion."""
        
        # Collect all document scores
        doc_scores = {}
        doc_info = {}
        
        for turn_results in all_results:
            for rank, doc in enumerate(turn_results):
                doc_id = doc["document_id"]
                rrf_score = 1.0 / (self.k + rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_info[doc_id] = doc
                
                doc_scores[doc_id] += rrf_score
        
        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in sorted_docs:
            results.append({
                "document_id": doc_id,
                "score": score,
                "text": doc_info[doc_id].get("text", ""),
                "title": doc_info[doc_id].get("title", "")
            })
        
        return results
    
    def _fuse_results_max(self, all_results: List[List[Dict]], top_k: int) -> List[Dict]:
        """Fuse results using max score."""
        doc_scores = {}
        doc_info = {}
        
        for turn_results in all_results:
            for doc in turn_results:
                doc_id = doc["document_id"]
                score = doc["score"]
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score
                    doc_info[doc_id] = doc
                else:
                    doc_scores[doc_id] = max(doc_scores[doc_id], score)
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            results.append({
                "document_id": doc_id,
                "score": score,
                "text": doc_info[doc_id].get("text", ""),
                "title": doc_info[doc_id].get("title", "")
            })
        
        return results
    
    def _fuse_results_mean(self, all_results: List[List[Dict]], top_k: int) -> List[Dict]:
        """Fuse results using mean score."""
        doc_scores = {}
        doc_counts = {}
        doc_info = {}
        
        for turn_results in all_results:
            for doc in turn_results:
                doc_id = doc["document_id"]
                score = doc["score"]
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_counts[doc_id] = 0
                    doc_info[doc_id] = doc
                
                doc_scores[doc_id] += score
                doc_counts[doc_id] += 1
        
        # Calculate mean
        for doc_id in doc_scores:
            doc_scores[doc_id] /= doc_counts[doc_id]
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            results.append({
                "document_id": doc_id,
                "score": score,
                "text": doc_info[doc_id].get("text", ""),
                "title": doc_info[doc_id].get("title", "")
            })
        
        return results
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multi-query approach.
        
        Args:
            query: Multi-turn query text
            top_k: Number of documents to return
            
        Returns:
            Fused retrieval results
        """
        # Extract individual turns
        turns = self._extract_turns(query)
        
        if len(turns) == 1:
            # Single turn, just use base retriever
            return self.base_retriever.retrieve(turns[0], top_k=top_k)
        
        # Retrieve for each turn
        all_results = []
        for turn in turns:
            turn_results = self.base_retriever.retrieve(turn, top_k=top_k * 2)
            all_results.append(turn_results)
        
        # Fuse results
        if self.fusion_method == "rrf":
            return self._fuse_results_rrf(all_results, top_k)
        elif self.fusion_method == "max":
            return self._fuse_results_max(all_results, top_k)
        elif self.fusion_method == "mean":
            return self._fuse_results_mean(all_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

