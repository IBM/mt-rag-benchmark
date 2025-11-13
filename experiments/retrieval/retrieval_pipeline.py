"""
Main retrieval pipeline for Task A.
"""

import json
import os
import sys
import argparse
from typing import Dict, List
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_retriever import BaseRetriever
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever
from corpus_loader import (
    load_corpus, load_queries, get_corpus_path, get_queries_path,
    get_collection_name, DOMAIN_NAMES, extract_query_text
)
from query_processor import QueryRewriter
from enhanced_query_processor import EnhancedQueryRewriter
from reranker import Reranker, RetrievalReranker
from multi_query_retriever import MultiQueryRetriever


def create_retriever(retriever_type: str, collection_name: str, 
                    use_reranker: bool = False, use_multi_query: bool = False,
                    **kwargs) -> BaseRetriever:
    """
    Create a retriever instance.
    
    Args:
        retriever_type: Type of retriever (bm25, dense, hybrid)
        collection_name: Collection name
        **kwargs: Additional arguments for retriever
        
    Returns:
        Retriever instance
    """
    if retriever_type == "bm25":
        return BM25Retriever(collection_name, 
                           k1=kwargs.get("bm25_k1", 1.5),
                           b=kwargs.get("bm25_b", 0.75))
    elif retriever_type == "dense":
        return DenseRetriever(collection_name,
                            model_name=kwargs.get("dense_model", "BAAI/bge-base-en-v1.5"),
                            device=kwargs.get("device", None),
                            batch_size=kwargs.get("batch_size", 32))
    elif retriever_type == "hybrid":
        retriever = HybridRetriever(collection_name,
                             bm25_k1=kwargs.get("bm25_k1", 1.5),
                             bm25_b=kwargs.get("bm25_b", 0.75),
                             dense_model=kwargs.get("dense_model", "BAAI/bge-base-en-v1.5"),
                             alpha=kwargs.get("alpha", 0.5),
                             fusion_method=kwargs.get("fusion_method", "rrf"))
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    # Wrap with multi-query if requested
    if use_multi_query:
        retriever = MultiQueryRetriever(
            retriever,
            fusion_method=kwargs.get("multi_query_fusion", "rrf"),
            k=kwargs.get("multi_query_k", 60)
        )
    
    # Wrap with re-ranker if requested
    if use_reranker:
        reranker = Reranker(
            model_name=kwargs.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=kwargs.get("device", None),
            batch_size=kwargs.get("reranker_batch_size", 32)
        )
        retriever = RetrievalReranker(
            retriever,
            reranker=reranker,
            initial_top_k=kwargs.get("reranker_initial_k", 100),
            final_top_k=kwargs.get("top_k", 10)
        )
    
    return retriever


def run_retrieval(domain: str, retriever: BaseRetriever, 
                 queries: Dict[str, str], query_rewriter: QueryRewriter = None,
                 top_k: int = 10, output_file: str = None) -> List[Dict]:
    """
    Run retrieval on queries.
    
    Args:
        domain: Domain name
        retriever: Retriever instance
        queries: Dictionary of query_id -> query_text
        query_rewriter: Optional query rewriter
        top_k: Number of documents to retrieve
        output_file: Optional output file path
        
    Returns:
        List of formatted results
    """
    collection_name = get_collection_name(domain)
    results = []
    
    print(f"\nRetrieving for {len(queries)} queries...")
    for query_id, query_text in tqdm(queries.items(), desc="Retrieving"):
        # Rewrite query if rewriter provided
        if query_rewriter:
            processed_query = query_rewriter.rewrite(query_text)
        else:
            processed_query = extract_query_text(query_text)
        
        # Retrieve
        retrieved_docs = retriever.retrieve(processed_query, top_k=top_k)
        
        # Format result
        result = retriever.format_results(query_id, retrieved_docs, collection_name)
        results.append(result)
    
    # Save results if output file provided
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieval pipeline for Task A")
    
    # Data arguments
    parser.add_argument("--domain", type=str, required=True, 
                       choices=DOMAIN_NAMES, help="Domain name")
    parser.add_argument("--corpus_path", type=str, default=None,
                       help="Path to corpus file (default: auto-detect)")
    parser.add_argument("--queries_path", type=str, default=None,
                       help="Path to queries file (default: auto-detect)")
    parser.add_argument("--query_type", type=str, default="lastturn",
                       choices=["lastturn", "questions", "rewrite"],
                       help="Query type to use")
    
    # Retriever arguments
    parser.add_argument("--retriever", type=str, required=True,
                       choices=["bm25", "dense", "hybrid"],
                       help="Retriever type")
    parser.add_argument("--dense_model", type=str, default="BAAI/bge-base-en-v1.5",
                       help="Dense model name")
    parser.add_argument("--bm25_k1", type=float, default=1.5,
                       help="BM25 parameter k1")
    parser.add_argument("--bm25_b", type=float, default=0.75,
                       help="BM25 parameter b")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for dense retrieval in hybrid (only for weighted fusion)")
    parser.add_argument("--fusion_method", type=str, default="rrf",
                       choices=["rrf", "weighted"],
                       help="Fusion method for hybrid retriever")
    
    # Query processing
    parser.add_argument("--query_rewrite", type=str, default=None,
                       choices=["last_turn", "all_turns", "expand", "full", 
                               "contextual", "weighted", "key_terms"],
                       help="Query rewriting method")
    
    # Advanced features
    parser.add_argument("--use_reranker", action="store_true",
                       help="Use re-ranker to improve results")
    parser.add_argument("--reranker_model", type=str, 
                       default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                       help="Re-ranker model name")
    parser.add_argument("--reranker_initial_k", type=int, default=100,
                       help="Number of docs to retrieve before re-ranking")
    parser.add_argument("--reranker_batch_size", type=int, default=32,
                       help="Batch size for re-ranker")
    
    parser.add_argument("--use_multi_query", action="store_true",
                       help="Use multi-query retrieval (retrieve for each turn)")
    parser.add_argument("--multi_query_fusion", type=str, default="rrf",
                       choices=["rrf", "max", "mean"],
                       help="Fusion method for multi-query")
    parser.add_argument("--multi_query_k", type=int, default=60,
                       help="RRF parameter for multi-query fusion")
    
    # Retrieval arguments
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of documents to retrieve")
    
    # Output
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file path for results")
    
    # Other
    parser.add_argument("--device", type=str, default=None,
                       help="Device for dense retrieval (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for dense retrieval")
    
    args = parser.parse_args()
    
    # Get paths
    if args.corpus_path is None:
        args.corpus_path = get_corpus_path(args.domain)
    if args.queries_path is None:
        args.queries_path = get_queries_path(args.domain, args.query_type)
    
    # Load data
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    corpus = load_corpus(args.corpus_path)
    queries = load_queries(args.queries_path)
    
    # Create retriever
    print("\n" + "=" * 60)
    print(f"Creating {args.retriever} retriever...")
    if args.use_reranker:
        print("  + Re-ranker enabled")
    if args.use_multi_query:
        print("  + Multi-query enabled")
    print("=" * 60)
    collection_name = get_collection_name(args.domain)
    retriever = create_retriever(
        args.retriever,
        collection_name,
        use_reranker=args.use_reranker,
        use_multi_query=args.use_multi_query,
        dense_model=args.dense_model,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        alpha=args.alpha,
        fusion_method=args.fusion_method,
        device=args.device,
        batch_size=args.batch_size,
        reranker_model=args.reranker_model,
        reranker_initial_k=args.reranker_initial_k,
        reranker_batch_size=args.reranker_batch_size,
        multi_query_fusion=args.multi_query_fusion,
        multi_query_k=args.multi_query_k,
        top_k=args.top_k
    )
    
    # Index corpus
    print("\n" + "=" * 60)
    print("Indexing corpus...")
    print("=" * 60)
    retriever.index(corpus)
    
    # Create query rewriter if specified
    query_rewriter = None
    if args.query_rewrite:
        # Use enhanced rewriter for advanced methods
        if args.query_rewrite in ["contextual", "weighted", "key_terms", "expanded"]:
            query_rewriter = EnhancedQueryRewriter(method=args.query_rewrite)
        else:
            query_rewriter = QueryRewriter(method=args.query_rewrite)
        print(f"\nUsing query rewriting: {args.query_rewrite}")
    
    # Run retrieval
    print("\n" + "=" * 60)
    print("Running retrieval...")
    print("=" * 60)
    results = run_retrieval(
        args.domain,
        retriever,
        queries,
        query_rewriter=query_rewriter,
        top_k=args.top_k,
        output_file=args.output_file
    )
    
    print("\n" + "=" * 60)
    print("Retrieval complete!")
    print(f"Retrieved {len(results)} queries")
    print(f"Results saved to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

