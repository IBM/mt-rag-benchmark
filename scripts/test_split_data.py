#!/usr/bin/env python3
"""
Test script to verify that split data is working correctly.

Checks:
1. No overlap between train/dev/test splits
2. All queries from same conversation are in same split
3. Qrels match queries in each split
4. Split ratios are approximately correct
5. All original queries are accounted for
"""

import json
import os
import sys
from collections import defaultdict
import pandas as pd


def extract_conversation_id(query_id: str) -> str:
    """Extract conversation identifier from query ID."""
    if '<::>' in query_id:
        return query_id.split('<::>')[0]
    return query_id.split('_')[0] if '_' in query_id else query_id


def load_queries(queries_path: str) -> dict:
    """Load queries from JSONL file."""
    queries = {}
    if not os.path.exists(queries_path):
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
        return pd.DataFrame(columns=['query-id', 'corpus-id', 'score'])
    
    return pd.read_csv(qrels_path, sep='\t', dtype=str)


def test_domain_splits(domain: str, query_type: str, base_dir: str):
    """Test splits for a single domain and query type."""
    print(f"\n{'='*60}")
    print(f"Testing {domain} - {query_type}")
    print(f"{'='*60}")
    
    # Load original data
    original_queries_path = os.path.join(
        base_dir.replace('_split', ''), 
        domain, 
        f"{domain}_{query_type}.jsonl"
    )
    original_qrels_path = os.path.join(
        base_dir.replace('_split', ''), 
        domain, 
        "qrels", 
        "dev.tsv"
    )
    
    original_queries = load_queries(original_queries_path)
    original_qrels = load_qrels(original_qrels_path)
    
    print(f"Original: {len(original_queries)} queries, {len(original_qrels)} qrel entries")
    
    # Load split data
    train_queries = load_queries(
        os.path.join(base_dir, domain, f"{domain}_{query_type}_train.jsonl")
    )
    dev_queries = load_queries(
        os.path.join(base_dir, domain, f"{domain}_{query_type}_dev.jsonl")
    )
    test_queries = load_queries(
        os.path.join(base_dir, domain, f"{domain}_{query_type}_test.jsonl")
    )
    
    train_qrels = load_qrels(os.path.join(base_dir, domain, "qrels", "train.tsv"))
    dev_qrels = load_qrels(os.path.join(base_dir, domain, "qrels", "dev.tsv"))
    test_qrels = load_qrels(os.path.join(base_dir, domain, "qrels", "test.tsv"))
    
    print(f"Train: {len(train_queries)} queries, {len(train_qrels)} qrel entries")
    print(f"Dev: {len(dev_queries)} queries, {len(dev_qrels)} qrel entries")
    print(f"Test: {len(test_queries)} queries, {len(test_qrels)} qrel entries")
    
    # Test 1: Check total queries match
    total_split = len(train_queries) + len(dev_queries) + len(test_queries)
    print(f"\n✓ Test 1: Total queries match")
    print(f"  Original: {len(original_queries)}, Split total: {total_split}")
    if len(original_queries) == total_split:
        print(f"  ✅ PASS: All queries accounted for")
    else:
        print(f"  ❌ FAIL: Missing {len(original_queries) - total_split} queries")
        return False
    
    # Test 2: Check no overlap between splits
    train_ids = set(train_queries.keys())
    dev_ids = set(dev_queries.keys())
    test_ids = set(test_queries.keys())
    
    overlap_train_dev = train_ids & dev_ids
    overlap_train_test = train_ids & test_ids
    overlap_dev_test = dev_ids & test_ids
    
    print(f"\n✓ Test 2: No overlap between splits")
    if not overlap_train_dev and not overlap_train_test and not overlap_dev_test:
        print(f"  ✅ PASS: No query overlap between splits")
    else:
        print(f"  ❌ FAIL: Found overlaps:")
        if overlap_train_dev:
            print(f"    Train-Dev: {len(overlap_train_dev)} queries")
        if overlap_train_test:
            print(f"    Train-Test: {len(overlap_train_test)} queries")
        if overlap_dev_test:
            print(f"    Dev-Test: {len(overlap_dev_test)} queries")
        return False
    
    # Test 3: Check conversation-level consistency
    print(f"\n✓ Test 3: Conversation-level consistency")
    
    # Group queries by conversation
    train_convs = defaultdict(list)
    dev_convs = defaultdict(list)
    test_convs = defaultdict(list)
    
    for qid in train_ids:
        conv_id = extract_conversation_id(qid)
        train_convs[conv_id].append(qid)
    
    for qid in dev_ids:
        conv_id = extract_conversation_id(qid)
        dev_convs[conv_id].append(qid)
    
    for qid in test_ids:
        conv_id = extract_conversation_id(qid)
        test_convs[conv_id].append(qid)
    
    # Check for conversations split across multiple sets
    all_conv_ids = set(train_convs.keys()) | set(dev_convs.keys()) | set(test_convs.keys())
    split_conversations = []
    
    for conv_id in all_conv_ids:
        in_train = conv_id in train_convs
        in_dev = conv_id in dev_convs
        in_test = conv_id in test_convs
        
        count = sum([in_train, in_dev, in_test])
        if count > 1:
            split_conversations.append(conv_id)
    
    if not split_conversations:
        print(f"  ✅ PASS: All conversations in single split")
        print(f"    Train: {len(train_convs)} conversations")
        print(f"    Dev: {len(dev_convs)} conversations")
        print(f"    Test: {len(test_convs)} conversations")
    else:
        print(f"  ❌ FAIL: {len(split_conversations)} conversations split across multiple sets")
        print(f"    Examples: {split_conversations[:5]}")
        return False
    
    # Test 4: Check qrels match queries
    print(f"\n✓ Test 4: Qrels match queries")
    
    train_qrel_queries = set(train_qrels['query-id'].unique()) if len(train_qrels) > 0 else set()
    dev_qrel_queries = set(dev_qrels['query-id'].unique()) if len(dev_qrels) > 0 else set()
    test_qrel_queries = set(test_qrels['query-id'].unique()) if len(test_qrels) > 0 else set()
    
    train_mismatch = train_qrel_queries - train_ids
    dev_mismatch = dev_qrel_queries - dev_ids
    test_mismatch = test_qrel_queries - test_ids
    
    if not train_mismatch and not dev_mismatch and not test_mismatch:
        print(f"  ✅ PASS: All qrels match queries in their split")
    else:
        print(f"  ❌ FAIL: Qrels reference queries not in split:")
        if train_mismatch:
            print(f"    Train: {len(train_mismatch)} mismatched qrels")
        if dev_mismatch:
            print(f"    Dev: {len(dev_mismatch)} mismatched qrels")
        if test_mismatch:
            print(f"    Test: {len(test_mismatch)} mismatched qrels")
        return False
    
    # Test 5: Check split ratios
    print(f"\n✓ Test 5: Split ratios")
    total = len(original_queries)
    train_ratio = len(train_queries) / total if total > 0 else 0
    dev_ratio = len(dev_queries) / total if total > 0 else 0
    test_ratio = len(test_queries) / total if total > 0 else 0
    
    print(f"  Train: {train_ratio:.1%} ({len(train_queries)}/{total})")
    print(f"  Dev: {dev_ratio:.1%} ({len(dev_queries)}/{total})")
    print(f"  Test: {test_ratio:.1%} ({len(test_queries)}/{total})")
    
    # Check if ratios are reasonable (within 5% of expected)
    expected_train = 0.7
    expected_dev = 0.15
    expected_test = 0.15
    
    train_ok = abs(train_ratio - expected_train) < 0.05
    dev_ok = abs(dev_ratio - expected_dev) < 0.05
    test_ok = abs(test_ratio - expected_test) < 0.05
    
    if train_ok and dev_ok and test_ok:
        print(f"  ✅ PASS: Ratios are approximately correct (~70/15/15)")
    else:
        print(f"  ⚠️  WARNING: Ratios deviate from expected (this is OK if splitting by conversations)")
    
    # Test 6: Check qrel totals match
    print(f"\n✓ Test 6: Qrel totals")
    total_split_qrels = len(train_qrels) + len(dev_qrels) + len(test_qrels)
    print(f"  Original: {len(original_qrels)} qrel entries")
    print(f"  Split total: {total_split_qrels} qrel entries")
    
    if len(original_qrels) == total_split_qrels:
        print(f"  ✅ PASS: All qrel entries accounted for")
    else:
        diff = len(original_qrels) - total_split_qrels
        print(f"  ⚠️  WARNING: {abs(diff)} qrel entries difference (may be due to query filtering)")
    
    return True


def main():
    base_dir = "human/retrieval_tasks_split"
    
    if not os.path.exists(base_dir):
        print(f"Error: Split directory not found: {base_dir}")
        print("Please run split_passage_data.py first")
        sys.exit(1)
    
    print("="*60)
    print("Testing Split Data Integrity")
    print("="*60)
    
    domains = ["clapnq", "cloud", "fiqa", "govt"]
    query_types = ["lastturn", "rewrite", "questions"]
    
    all_passed = True
    
    for domain in domains:
        for query_type in query_types:
            try:
                passed = test_domain_splits(domain, query_type, base_dir)
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"\n❌ ERROR testing {domain}/{query_type}: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

