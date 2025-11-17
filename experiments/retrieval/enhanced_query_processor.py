"""
Enhanced query processing with better expansion and rewriting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from typing import List, Dict, Optional
from query_processor import extract_last_turn, extract_all_user_turns


def extract_key_terms(query: str) -> List[str]:
    """
    Extract key terms from query (remove stopwords, keep important words).
    
    Args:
        query: Query text
        
    Returns:
        List of key terms
    """
    # Simple stopwords list (can be expanded)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', query.lower())
    key_terms = [w for w in words if w not in stopwords and len(w) > 2]
    return key_terms


def expand_query_with_synonyms(query: str) -> str:
    """
    Expand query with synonyms (basic implementation).
    Can be enhanced with WordNet or LLM-based expansion.
    
    Args:
        query: Original query
        
    Returns:
        Expanded query
    """
    # Basic synonym expansion (can be enhanced)
    synonym_map = {
        'how': ['method', 'way', 'process'],
        'what': ['definition', 'meaning', 'explanation'],
        'why': ['reason', 'cause', 'purpose'],
        'when': ['time', 'date', 'period'],
        'where': ['location', 'place', 'position'],
        'who': ['person', 'individual', 'people'],
    }
    
    expanded_terms = []
    words = query.lower().split()
    
    for word in words:
        expanded_terms.append(word)
        if word in synonym_map:
            expanded_terms.extend(synonym_map[word][:1])  # Add one synonym
    
    return ' '.join(expanded_terms)


def create_contextual_query(query_text: str, method: str = "smart") -> str:
    """
    Create contextual query from multi-turn conversation.
    
    Args:
        query_text: Full conversation text with speaker tags
        method: Method for creating query:
            - "smart": Extract last turn + key terms from previous turns
            - "weighted": Weight recent turns more heavily
            - "full": Use all conversation context
            
    Returns:
        Contextual query
    """
    lines = query_text.split('\n')
    user_turns = []
    agent_turns = []
    
    for line in lines:
        if line.strip().startswith('|user|:'):
            user_turns.append(re.sub(r'\|user\|:\s*', '', line).strip())
        elif line.strip().startswith('|agent|:'):
            agent_turns.append(re.sub(r'\|agent\|:\s*', '', line).strip())
    
    if not user_turns:
        return query_text.strip()
    
    if method == "smart":
        # Use last turn + key terms from previous turns
        last_turn = user_turns[-1]
        if len(user_turns) > 1:
            # Extract key terms from previous turns
            previous_context = ' '.join(user_turns[:-1])
            key_terms = extract_key_terms(previous_context)
            # Combine
            if key_terms:
                return f"{last_turn} {' '.join(key_terms[:5])}"
        return last_turn
    
    elif method == "weighted":
        # Weight recent turns more
        if len(user_turns) == 1:
            return user_turns[0]
        # Last turn gets full weight, previous turns get reduced weight
        weighted_query = user_turns[-1]  # Last turn
        for i, turn in enumerate(user_turns[:-1][::-1]):  # Reverse order
            weight = 0.5 / (i + 1)  # Decreasing weight
            if weight > 0.1:  # Only include if weight is significant
                weighted_query = f"{turn} {weighted_query}"
        return weighted_query
    
    elif method == "full":
        # Use all user turns
        return ' '.join(user_turns)
    
    else:
        return user_turns[-1]


def rewrite_query_advanced(query_text: str, method: str = "contextual") -> str:
    """
    Advanced query rewriting with multiple strategies.
    
    Args:
        query_text: Original query text
        method: Rewriting method:
            - "contextual": Smart contextual query
            - "expanded": Query with synonym expansion
            - "key_terms": Extract and use key terms
            - "weighted": Weighted multi-turn query
            
    Returns:
        Rewritten query
    """
    if method == "contextual":
        return create_contextual_query(query_text, method="smart")
    elif method == "expanded":
        last_turn = extract_last_turn(query_text)
        return expand_query_with_synonyms(last_turn)
    elif method == "key_terms":
        all_turns = extract_all_user_turns(query_text)
        key_terms = extract_key_terms(all_turns)
        return ' '.join(key_terms) if key_terms else all_turns
    elif method == "weighted":
        return create_contextual_query(query_text, method="weighted")
    else:
        return extract_last_turn(query_text)


class EnhancedQueryRewriter:
    """Enhanced query rewriter with multiple strategies."""
    
    def __init__(self, method: str = "contextual"):
        """
        Initialize enhanced query rewriter.
        
        Args:
            method: Rewriting method (contextual, expanded, key_terms, weighted, 
                   last_turn, all_turns, full)
        """
        self.method = method
    
    def rewrite(self, query_text: str) -> str:
        """
        Rewrite a query using the specified method.
        
        Args:
            query_text: Original query text
            
        Returns:
            Rewritten query
        """
        if self.method in ["contextual", "expanded", "key_terms", "weighted"]:
            return rewrite_query_advanced(query_text, self.method)
        else:
            # Fall back to basic methods
            from query_processor import rewrite_query_with_context
            return rewrite_query_with_context(query_text, self.method)
    
    def batch_rewrite(self, queries: List[str]) -> List[str]:
        """
        Rewrite multiple queries.
        
        Args:
            queries: List of query texts
            
        Returns:
            List of rewritten queries
        """
        return [self.rewrite(q) for q in queries]

