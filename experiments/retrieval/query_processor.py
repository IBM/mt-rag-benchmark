"""
Query processing and rewriting utilities for multi-turn conversations.
"""

import re
from typing import List, Dict


def extract_last_turn(query_text: str) -> str:
    """
    Extract only the last user turn from a multi-turn query.
    
    Args:
        query_text: Query text with speaker tags
        
    Returns:
        Last user turn text
    """
    lines = query_text.split('\n')
    user_turns = [line for line in lines if line.strip().startswith('|user|:')]
    if user_turns:
        last_turn = user_turns[-1]
        # Remove speaker tag
        return re.sub(r'\|user\|:\s*', '', last_turn).strip()
    return query_text.strip()


def extract_all_user_turns(query_text: str) -> str:
    """
    Extract all user turns from a multi-turn query.
    
    Args:
        query_text: Query text with speaker tags
        
    Returns:
        All user turns combined
    """
    lines = query_text.split('\n')
    user_turns = [re.sub(r'\|user\|:\s*', '', line).strip() 
                  for line in lines if line.strip().startswith('|user|:')]
    return ' '.join(user_turns)


def simple_query_expansion(query: str, include_context: bool = True) -> str:
    """
    Simple query expansion by adding synonyms or related terms.
    This is a basic implementation - can be enhanced with LLM-based expansion.
    
    Args:
        query: Original query
        include_context: Whether to include context words
        
    Returns:
        Expanded query
    """
    # Basic expansion: add question words if missing
    question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which']
    query_lower = query.lower()
    
    # If query doesn't start with a question word, it might benefit from expansion
    # This is a simple heuristic - can be improved
    expanded = query
    
    # Add common related terms (very basic - should use proper expansion)
    if 'definition' in query_lower or 'meaning' in query_lower:
        expanded = f"{query} explanation"
    elif 'how' in query_lower and 'work' in query_lower:
        expanded = f"{query} process steps"
    
    return expanded


def rewrite_query_with_context(query_text: str, method: str = "last_turn") -> str:
    """
    Rewrite query using different strategies.
    
    Args:
        query_text: Original query text
        method: Rewriting method:
            - "last_turn": Use only last user turn
            - "all_turns": Use all user turns
            - "expand": Simple expansion
            - "full": Use full conversation context
            
    Returns:
        Rewritten query
    """
    if method == "last_turn":
        return extract_last_turn(query_text)
    elif method == "all_turns":
        return extract_all_user_turns(query_text)
    elif method == "expand":
        last_turn = extract_last_turn(query_text)
        return simple_query_expansion(last_turn)
    elif method == "full":
        # Remove speaker tags but keep full context
        cleaned = re.sub(r'\|(user|agent)\|:\s*', '', query_text)
        return cleaned.strip()
    else:
        return query_text.strip()


class QueryRewriter:
    """Query rewriter for multi-turn conversations."""
    
    def __init__(self, method: str = "last_turn"):
        """
        Initialize query rewriter.
        
        Args:
            method: Rewriting method (last_turn, all_turns, expand, full)
        """
        self.method = method
    
    def rewrite(self, query_text: str) -> str:
        """
        Rewrite a query.
        
        Args:
            query_text: Original query text
            
        Returns:
            Rewritten query
        """
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

