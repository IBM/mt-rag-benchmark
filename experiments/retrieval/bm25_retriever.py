"""
BM25 lexical retriever implementation.
"""

import json
from typing import Dict, List, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_retriever import BaseRetriever

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class BM25Retriever(BaseRetriever):
    """BM25 lexical retriever."""
    
    def __init__(self, collection_name: str, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            collection_name: Collection name
            k1: BM25 parameter k1 (term frequency saturation)
            b: BM25 parameter b (length normalization)
        """
        super().__init__(collection_name)
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = []
        self.documents = {}
        self.stop_words = set(stopwords.words('english'))
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text."""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        return tokens
    
    def index(self, corpus: Dict[str, Dict[str, str]]):
        """
        Index corpus using BM25.
        
        Args:
            corpus: Dictionary mapping document_id to {"text": ..., "title": ...}
        """
        print(f"Indexing {len(corpus)} documents for BM25...")
        
        self.doc_ids = []
        self.documents = {}
        tokenized_corpus = []
        
        for doc_id, doc in corpus.items():
            # Combine title and text
            full_text = ""
            if doc.get("title"):
                full_text += doc["title"] + " "
            full_text += doc.get("text", "")
            
            # Tokenize
            tokens = self._tokenize(full_text)
            tokenized_corpus.append(tokens)
            self.doc_ids.append(doc_id)
            self.documents[doc_id] = doc
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.indexed = True
        print(f"Indexed {len(self.doc_ids)} documents.")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.indexed:
            raise ValueError("Corpus must be indexed before retrieval. Call index() first.")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
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

