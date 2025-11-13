"""
Utilities for loading and processing corpora.
"""

import json
import os
import zipfile
from typing import Dict, List
from tqdm import tqdm


# Collection name mapping
COLLECTION_MAPPING = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "govt": "mt-rag-govt-elser-512-100-20240611"
}

DOMAIN_NAMES = ["clapnq", "cloud", "fiqa", "govt"]


def load_corpus(corpus_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load corpus from JSONL file or zip file containing JSONL.
    
    Args:
        corpus_path: Path to corpus JSONL file or zip file
        
    Returns:
        Dictionary mapping document_id to {"text": ..., "title": ...}
    """
    corpus = {}
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    print(f"Loading corpus from {corpus_path}...")
    
    # Handle zip files
    if corpus_path.endswith('.zip'):
        with zipfile.ZipFile(corpus_path, 'r') as z:
            # Get the JSONL file from zip (usually just one file)
            files = [f for f in z.namelist() if f.endswith('.jsonl')]
            if not files:
                raise ValueError(f"No JSONL file found in zip: {corpus_path}")
            jsonl_file = files[0]
            
            # Read from zip
            with z.open(jsonl_file) as f:
                for line in tqdm(f, desc="Loading documents"):
                    doc = json.loads(line.decode('utf-8').strip())
                    doc_id = doc.get("_id")
                    if doc_id:
                        corpus[doc_id] = {
                            "text": doc.get("text", ""),
                            "title": doc.get("title", ""),
                            "url": doc.get("url", "")
                        }
    else:
        # Regular JSONL file
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading documents"):
                doc = json.loads(line.strip())
                doc_id = doc.get("_id")
                if doc_id:
                    corpus[doc_id] = {
                        "text": doc.get("text", ""),
                        "title": doc.get("title", ""),
                        "url": doc.get("url", "")
                    }
    
    print(f"Loaded {len(corpus)} documents.")
    return corpus


def load_queries(queries_path: str) -> Dict[str, str]:
    """
    Load queries from JSONL file.
    
    Args:
        queries_path: Path to queries JSONL file
        
    Returns:
        Dictionary mapping query_id to query text
    """
    queries = {}
    
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    
    print(f"Loading queries from {queries_path}...")
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            query_id = query.get("_id")
            query_text = query.get("text", "")
            if query_id:
                queries[query_id] = query_text
    
    print(f"Loaded {len(queries)} queries.")
    return queries


def get_corpus_path(domain: str, base_dir: str = None) -> str:
    """
    Get corpus file path for a domain.
    Tries .jsonl first, then .jsonl.zip
    
    Args:
        domain: Domain name (clapnq, cloud, fiqa, govt)
        base_dir: Base directory for corpora (default: auto-detect from project root)
        
    Returns:
        Path to corpus file
    """
    domain_lower = domain.lower()
    if domain_lower == "clapnq":
        base_name = "clapnq"
    elif domain_lower == "cloud":
        base_name = "cloud"
    elif domain_lower == "fiqa":
        base_name = "fiqa"
    elif domain_lower == "govt":
        base_name = "govt"
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Auto-detect project root if base_dir not provided
    if base_dir is None:
        # Try to find project root (directory containing 'corpora' folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root (from experiments/retrieval to project root)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        base_dir = os.path.join(project_root, "corpora", "passage_level")
    
    # Try .jsonl first, then .jsonl.zip
    jsonl_path = os.path.join(base_dir, f"{base_name}.jsonl")
    zip_path = os.path.join(base_dir, f"{base_name}.jsonl.zip")
    
    if os.path.exists(jsonl_path):
        return jsonl_path
    elif os.path.exists(zip_path):
        return zip_path
    else:
        raise FileNotFoundError(f"Corpus file not found for {domain}. Tried: {jsonl_path}, {zip_path}")


def get_queries_path(domain: str, query_type: str = "lastturn", 
                    base_dir: str = None) -> str:
    """
    Get queries file path for a domain and query type.
    
    Args:
        domain: Domain name (clapnq, cloud, fiqa, govt)
        query_type: Query type (lastturn, questions, rewrite)
        base_dir: Base directory for queries (default: auto-detect from project root)
        
    Returns:
        Path to queries file
    """
    domain_lower = domain.lower()
    filename = f"{domain_lower}_{query_type}.jsonl"
    
    # Auto-detect project root if base_dir not provided
    if base_dir is None:
        # Try to find project root (directory containing 'human' folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root (from experiments/retrieval to project root)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        base_dir = os.path.join(project_root, "human", "retrieval_tasks")
    
    return os.path.join(base_dir, domain_lower, filename)


def get_collection_name(domain: str) -> str:
    """
    Get collection name for a domain.
    
    Args:
        domain: Domain name (clapnq, cloud, fiqa, govt)
        
    Returns:
        Collection name
    """
    domain_lower = domain.lower()
    return COLLECTION_MAPPING.get(domain_lower, f"mt-rag-{domain_lower}")


def extract_query_text(query_text: str, remove_speaker_tags: bool = True) -> str:
    """
    Extract and clean query text.
    
    Args:
        query_text: Raw query text (may contain speaker tags)
        remove_speaker_tags: Whether to remove |user|:|agent| tags
        
    Returns:
        Cleaned query text
    """
    if remove_speaker_tags:
        # Remove speaker tags like |user|: or |agent|:
        import re
        query_text = re.sub(r'\|(user|agent)\|:\s*', '', query_text)
        # Remove leading/trailing whitespace
        query_text = query_text.strip()
    
    return query_text

