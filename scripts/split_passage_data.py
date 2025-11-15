#!/usr/bin/env python3
"""
Split passage-level retrieval data into train/dev/test sets.

Best practices for retrieval task splitting:
1. Keep entire corpus intact (all passages available for all splits)
2. Split queries and qrels by conversation to avoid data leakage
3. Use stratified splitting to maintain domain/question type distribution
4. Standard split ratios: 70% train, 15% dev, 15% test (or 80/10/10)

Usage:
    python scripts/split_passage_data.py \
        --input_dir human/retrieval_tasks \
        --output_dir human/retrieval_tasks_split \
        --train_ratio 0.7 \
        --dev_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42
"""

import json
import os
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd


def extract_conversation_id(query_id: str) -> str:
    """
    Extract conversation identifier from query ID.
    
    Query IDs are in format: {author}_{timestamp}<::>{turn_number}
    Conversation ID is: {author}_{timestamp}
    """
    if '<::>' in query_id:
        return query_id.split('<::>')[0]
    # Fallback: use first part before any separator
    return query_id.split('_')[0] if '_' in query_id else query_id


def load_queries(queries_path: str) -> Dict[str, Dict]:
    """Load queries from JSONL file."""
    queries = {}
    if not os.path.exists(queries_path):
        print(f"Warning: Queries file not found: {queries_path}")
        return queries
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            query_id = query.get("_id")
            if query_id:
                queries[query_id] = query
    
    return queries


def load_qrels(qrels_path: str) -> pd.DataFrame:
    """Load qrels from TSV file."""
    if not os.path.exists(qrels_path):
        print(f"Warning: Qrels file not found: {qrels_path}")
        return pd.DataFrame(columns=['query-id', 'corpus-id', 'score'])
    
    return pd.read_csv(qrels_path, sep='\t', dtype=str)


def group_queries_by_conversation(queries: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Group query IDs by conversation ID."""
    conversation_queries = defaultdict(list)
    for query_id in queries.keys():
        conv_id = extract_conversation_id(query_id)
        conversation_queries[conv_id].append(query_id)
    return conversation_queries


def split_conversations(
    conversation_queries: Dict[str, List[str]],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split conversations into train/dev/test sets.
    
    Returns:
        Tuple of (train_conversations, dev_conversations, test_conversations)
    """
    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get list of conversation IDs
    conv_ids = list(conversation_queries.keys())
    random.seed(seed)
    random.shuffle(conv_ids)
    
    # Calculate split points
    n_total = len(conv_ids)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    
    # Split
    train_conv_ids = conv_ids[:n_train]
    dev_conv_ids = conv_ids[n_train:n_train + n_dev]
    test_conv_ids = conv_ids[n_train + n_dev:]
    
    return train_conv_ids, dev_conv_ids, test_conv_ids


def get_query_ids_from_conversations(
    conversation_queries: Dict[str, List[str]],
    conv_ids: List[str]
) -> Set[str]:
    """Get all query IDs from given conversation IDs."""
    query_ids = set()
    for conv_id in conv_ids:
        query_ids.update(conversation_queries[conv_id])
    return query_ids


def save_queries(queries: Dict[str, Dict], query_ids: Set[str], output_path: str):
    """Save queries to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for query_id in sorted(query_ids):
            if query_id in queries:
                f.write(json.dumps(queries[query_id], ensure_ascii=False) + '\n')


def save_qrels(qrels: pd.DataFrame, query_ids: Set[str], output_path: str):
    """Save qrels filtered by query IDs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter qrels to only include queries in the split
    filtered_qrels = qrels[qrels['query-id'].isin(query_ids)]
    filtered_qrels.to_csv(output_path, sep='\t', index=False)


def split_domain_data(
    input_dir: str,
    output_dir: str,
    domain: str,
    query_type: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int
):
    """Split data for a single domain and query type."""
    print(f"\nProcessing {domain} - {query_type}...")
    
    # Load queries and qrels
    queries_file = os.path.join(input_dir, domain, f"{domain}_{query_type}.jsonl")
    qrels_file = os.path.join(input_dir, domain, "qrels", "dev.tsv")
    
    queries = load_queries(queries_file)
    qrels = load_qrels(qrels_file)
    
    if not queries:
        print(f"  Skipping {domain}/{query_type} - no queries found")
        return
    
    print(f"  Loaded {len(queries)} queries, {len(qrels)} qrel entries")
    
    # Group queries by conversation
    conversation_queries = group_queries_by_conversation(queries)
    print(f"  Found {len(conversation_queries)} conversations")
    
    # Split conversations
    train_conv_ids, dev_conv_ids, test_conv_ids = split_conversations(
        conversation_queries, train_ratio, dev_ratio, test_ratio, seed
    )
    
    print(f"  Split: {len(train_conv_ids)} train, {len(dev_conv_ids)} dev, {len(test_conv_ids)} test conversations")
    
    # Get query IDs for each split
    train_query_ids = get_query_ids_from_conversations(conversation_queries, train_conv_ids)
    dev_query_ids = get_query_ids_from_conversations(conversation_queries, dev_conv_ids)
    test_query_ids = get_query_ids_from_conversations(conversation_queries, test_conv_ids)
    
    print(f"  Queries: {len(train_query_ids)} train, {len(dev_query_ids)} dev, {len(test_query_ids)} test")
    
    # Save train split
    train_queries_path = os.path.join(output_dir, domain, f"{domain}_{query_type}_train.jsonl")
    train_qrels_path = os.path.join(output_dir, domain, "qrels", "train.tsv")
    save_queries(queries, train_query_ids, train_queries_path)
    save_qrels(qrels, train_query_ids, train_qrels_path)
    print(f"  Saved train: {len(train_query_ids)} queries")
    
    # Save dev split
    dev_queries_path = os.path.join(output_dir, domain, f"{domain}_{query_type}_dev.jsonl")
    dev_qrels_path = os.path.join(output_dir, domain, "qrels", "dev.tsv")
    save_queries(queries, dev_query_ids, dev_queries_path)
    save_qrels(qrels, dev_query_ids, dev_qrels_path)
    print(f"  Saved dev: {len(dev_query_ids)} queries")
    
    # Save test split
    test_queries_path = os.path.join(output_dir, domain, f"{domain}_{query_type}_test.jsonl")
    test_qrels_path = os.path.join(output_dir, domain, "qrels", "test.tsv")
    save_queries(queries, test_query_ids, test_queries_path)
    save_qrels(qrels, test_query_ids, test_qrels_path)
    print(f"  Saved test: {len(test_query_ids)} queries")


def main():
    parser = argparse.ArgumentParser(
        description="Split passage-level retrieval data into train/dev/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 70/15/15 split
  python scripts/split_passage_data.py \\
      --input_dir human/retrieval_tasks \\
      --output_dir human/retrieval_tasks_split \\
      --train_ratio 0.7 --dev_ratio 0.15 --test_ratio 0.15

  # 80/10/10 split with specific seed
  python scripts/split_passage_data.py \\
      --input_dir human/retrieval_tasks \\
      --output_dir human/retrieval_tasks_split \\
      --train_ratio 0.8 --dev_ratio 0.1 --test_ratio 0.1 \\
      --seed 123

  # Split only specific domain
  python scripts/split_passage_data.py \\
      --input_dir human/retrieval_tasks \\
      --output_dir human/retrieval_tasks_split \\
      --domains clapnq \\
      --train_ratio 0.7 --dev_ratio 0.15 --test_ratio 0.15
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing retrieval tasks (e.g., human/retrieval_tasks)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for split data"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of data for training (default: 0.7)"
    )
    
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.15,
        help="Ratio of data for development/validation (default: 0.15)"
    )
    
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio of data for testing (default: 0.15)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["clapnq", "cloud", "fiqa", "govt"],
        help="Domains to process (default: all four domains)"
    )
    
    parser.add_argument(
        "--query_types",
        type=str,
        nargs="+",
        default=["lastturn", "rewrite", "questions"],
        help="Query types to process (default: all three types)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {total_ratio}")
    
    print("=" * 60)
    print("Passage-Level Data Splitter")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: Train={args.train_ratio:.1%}, Dev={args.dev_ratio:.1%}, Test={args.test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Query types: {', '.join(args.query_types)}")
    print("=" * 60)
    
    # Process each domain and query type
    for domain in args.domains:
        for query_type in args.query_types:
            try:
                split_domain_data(
                    args.input_dir,
                    args.output_dir,
                    domain,
                    query_type,
                    args.train_ratio,
                    args.dev_ratio,
                    args.test_ratio,
                    args.seed
                )
            except Exception as e:
                print(f"  Error processing {domain}/{query_type}: {e}")
                continue
    
    print("\n" + "=" * 60)
    print("Split complete!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    <domain>/")
    print(f"      <domain>_<query_type>_train.jsonl")
    print(f"      <domain>_<query_type>_dev.jsonl")
    print(f"      <domain>_<query_type>_test.jsonl")
    print(f"      qrels/")
    print(f"        train.tsv")
    print(f"        dev.tsv")
    print(f"        test.tsv")
    print("\nNote: The corpus files remain unchanged - use the original")
    print("      corpus files from corpora/passage_level/ for all splits.")


if __name__ == "__main__":
    main()

