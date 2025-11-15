#!/usr/bin/env python3
"""
Split document-level corpus into train/dev/test sets.

This script splits the document corpus files (not queries) into train/dev/test sets.
Useful for:
- Training document encoders
- Fine-tuning models on documents
- Creating separate document collections

Note: For retrieval tasks, you typically keep the corpus intact and split queries instead.
This script is for cases where you need to split the documents themselves.

Usage:
    python scripts/split_document_corpus.py \
        --input_dir corpora/document_level \
        --output_dir corpora/document_level_split \
        --train_ratio 0.7 \
        --dev_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42
"""

import json
import os
import argparse
import random
import zipfile
from typing import List, Dict
from tqdm import tqdm


def load_documents(corpus_path: str) -> List[Dict]:
    """
    Load documents from JSONL file or zip file containing JSONL.
    
    Args:
        corpus_path: Path to corpus JSONL file or zip file
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    print(f"Loading documents from {corpus_path}...")
    
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
                    documents.append(doc)
    else:
        # Regular JSONL file
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading documents"):
                doc = json.loads(line.strip())
                documents.append(doc)
    
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(
    documents: List[Dict],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split documents into train/dev/test sets.
    
    Args:
        documents: List of document dictionaries
        train_ratio: Ratio for training set
        dev_ratio: Ratio for development set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_docs, dev_docs, test_docs)
    """
    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle documents
    random.seed(seed)
    shuffled_docs = documents.copy()
    random.shuffle(shuffled_docs)
    
    # Calculate split points
    n_total = len(shuffled_docs)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    
    # Split
    train_docs = shuffled_docs[:n_train]
    dev_docs = shuffled_docs[n_train:n_train + n_dev]
    test_docs = shuffled_docs[n_train + n_dev:]
    
    return train_docs, dev_docs, test_docs


def save_documents(documents: List[Dict], output_path: str, compress: bool = False):
    """
    Save documents to JSONL file or zip file.
    
    Args:
        documents: List of document dictionaries
        output_path: Output file path
        compress: Whether to compress as zip file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if compress or output_path.endswith('.zip'):
        # Save as zip file
        if not output_path.endswith('.zip'):
            output_path += '.zip'
        
        jsonl_filename = os.path.basename(output_path).replace('.zip', '.jsonl')
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z:
            # Write to a temporary string first, then add to zip
            import io
            jsonl_content = io.StringIO()
            for doc in tqdm(documents, desc=f"Saving {os.path.basename(output_path)}"):
                jsonl_content.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            z.writestr(jsonl_filename, jsonl_content.getvalue())
    else:
        # Save as regular JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in tqdm(documents, desc=f"Saving {os.path.basename(output_path)}"):
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(documents)} documents to {output_path}")


def split_domain_corpus(
    input_dir: str,
    output_dir: str,
    domain: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    compress: bool
):
    """Split corpus for a single domain."""
    print(f"\nProcessing {domain}...")
    
    # Load corpus
    corpus_file = os.path.join(input_dir, f"{domain}.jsonl.zip")
    if not os.path.exists(corpus_file):
        # Try without .zip extension
        corpus_file = os.path.join(input_dir, f"{domain}.jsonl")
        if not os.path.exists(corpus_file):
            print(f"  Skipping {domain} - corpus file not found")
            return
    
    documents = load_documents(corpus_file)
    
    if not documents:
        print(f"  Skipping {domain} - no documents found")
        return
    
    # Split documents
    train_docs, dev_docs, test_docs = split_documents(
        documents, train_ratio, dev_ratio, test_ratio, seed
    )
    
    print(f"  Split: {len(train_docs)} train, {len(dev_docs)} dev, {len(test_docs)} test documents")
    
    # Save splits
    if compress:
        train_path = os.path.join(output_dir, f"{domain}_train.jsonl.zip")
        dev_path = os.path.join(output_dir, f"{domain}_dev.jsonl.zip")
        test_path = os.path.join(output_dir, f"{domain}_test.jsonl.zip")
    else:
        train_path = os.path.join(output_dir, f"{domain}_train.jsonl")
        dev_path = os.path.join(output_dir, f"{domain}_dev.jsonl")
        test_path = os.path.join(output_dir, f"{domain}_test.jsonl")
    
    save_documents(train_docs, train_path, compress)
    save_documents(dev_docs, dev_path, compress)
    save_documents(test_docs, test_path, compress)


def main():
    parser = argparse.ArgumentParser(
        description="Split document-level corpus into train/dev/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 70/15/15 split (compressed)
  python scripts/split_document_corpus.py \\
      --input_dir corpora/document_level \\
      --output_dir corpora/document_level_split \\
      --train_ratio 0.7 --dev_ratio 0.15 --test_ratio 0.15

  # 80/10/10 split (uncompressed)
  python scripts/split_document_corpus.py \\
      --input_dir corpora/document_level \\
      --output_dir corpora/document_level_split \\
      --train_ratio 0.8 --dev_ratio 0.1 --test_ratio 0.1 \\
      --no_compress

  # Split only specific domain
  python scripts/split_document_corpus.py \\
      --input_dir corpora/document_level \\
      --output_dir corpora/document_level_split \\
      --domains clapnq \\
      --train_ratio 0.7 --dev_ratio 0.15 --test_ratio 0.15
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing document corpus files (e.g., corpora/document_level)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for split corpus files"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of documents for training (default: 0.7)"
    )
    
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.15,
        help="Ratio of documents for development/validation (default: 0.15)"
    )
    
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio of documents for testing (default: 0.15)"
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
        "--compress",
        action="store_true",
        default=True,
        help="Compress output files as zip (default: True)"
    )
    
    parser.add_argument(
        "--no_compress",
        action="store_false",
        dest="compress",
        help="Don't compress output files"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {total_ratio}")
    
    print("=" * 60)
    print("Document-Level Corpus Splitter")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: Train={args.train_ratio:.1%}, Dev={args.dev_ratio:.1%}, Test={args.test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print(f"Compress output: {args.compress}")
    print(f"Domains: {', '.join(args.domains)}")
    print("=" * 60)
    
    # Process each domain
    for domain in args.domains:
        try:
            split_domain_corpus(
                args.input_dir,
                args.output_dir,
                domain,
                args.train_ratio,
                args.dev_ratio,
                args.test_ratio,
                args.seed,
                args.compress
            )
        except Exception as e:
            print(f"  Error processing {domain}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Split complete!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    <domain>_train.jsonl{'(.zip)' if args.compress else ''}")
    print(f"    <domain>_dev.jsonl{'(.zip)' if args.compress else ''}")
    print(f"    <domain>_test.jsonl{'(.zip)' if args.compress else ''}")
    print("\nNote: This splits the document corpus itself.")
    print("      For retrieval tasks, you typically keep the corpus intact")
    print("      and split queries instead (see split_passage_data.py).")


if __name__ == "__main__":
    main()

