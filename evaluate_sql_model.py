#!/usr/bin/env python3
"""
Evaluate SQL Bug Detection Model on NL2SQL-Bugs Benchmark

This script evaluates a trained SQL bug detection model on the NL2SQL-Bugs benchmark.
It downloads the benchmark data, preprocesses it, runs inference, and computes metrics.

NL2SQL-Bugs Benchmark:
    - 2,018 examples (999 incorrect, 1,019 correct)
    - 9 main error categories with 31 subcategories
    - Semantic errors in NL-to-SQL translation

Metrics:
    - Overall Accuracy
    - Precision, Recall, F1 (for buggy class)
    - Type-Specific Accuracy (per error category)
    - Confusion Matrix

Usage:
    python evaluate_sql_model.py \
        --model-path models/sql-bug-detector-mnr \
        --benchmark-data data/nl2sql_bugs/NL2SQL-Bugs.json \
        --threshold 0.5 \
        --output-dir results/nl2sql_bugs_eval
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from sql_utils import (
    infer_buggy_sql_lines_from_diff,
    get_sql_context_before,
    get_sql_context_after,
    get_sql_context_full,
)


def download_nl2sql_bugs(output_dir: Path):
    """
    Download NL2SQL-Bugs benchmark data.

    Args:
        output_dir: Directory to save downloaded data

    Note:
        This downloads from the GitHub repository.
        Users should manually download if this fails.
    """
    import urllib.request

    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        "NL2SQL-Bugs.json",
        "NL2SQL-Bugs-with-evidence.json",
    ]

    base_url = "https://raw.githubusercontent.com/HKUSTDial/NL2SQL-Bugs-Benchmark/main/data/"

    print("Downloading NL2SQL-Bugs benchmark...")

    for filename in files_to_download:
        url = base_url + filename
        output_path = output_dir / filename

        if output_path.exists():
            print(f"  {filename} already exists, skipping...")
            continue

        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"  ✓ Downloaded to {output_path}")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            print(f"  Please manually download from: {url}")


def load_nl2sql_bugs(data_path: Path) -> List[Dict[str, Any]]:
    """
    Load NL2SQL-Bugs benchmark data.

    Args:
        data_path: Path to NL2SQL-Bugs.json file

    Returns:
        List of benchmark entries
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark data not found: {data_path}")

    print(f"Loading benchmark data from: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} benchmark examples")

    return data


def preprocess_nl2sql_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single NL2SQL-Bugs entry.

    The exact structure depends on the benchmark format.
    This is a template that may need adjustment based on actual data.

    Args:
        entry: Raw benchmark entry

    Returns:
        Preprocessed entry with standardized fields
    """
    # NOTE: Adjust these field names based on actual NL2SQL-Bugs structure
    # Common fields might include: 'SQL', 'question', 'db_id', 'is_correct', 'error_type', etc.

    processed = {
        'query': entry.get('SQL', entry.get('query', '')),
        'question': entry.get('question', entry.get('nl_query', '')),
        'db_id': entry.get('db_id', ''),
        'is_correct': entry.get('is_correct', entry.get('label', True)),
        'error_type': entry.get('error_type', entry.get('category', 'unknown')),
        'original_entry': entry
    }

    return processed


def compute_line_similarity_scores(
    model: SentenceTransformer,
    query: str,
    context_type: str = 'full',
    context_size: int = 3
) -> List[Tuple[int, str, float]]:
    """
    Compute similarity scores for each line in a SQL query.

    Args:
        model: Trained SentenceTransformer model
        query: SQL query string
        context_type: 'before', 'after', or 'full'
        context_size: Number of context lines for before/after

    Returns:
        List of (line_index, line_text, similarity_score) tuples
    """
    lines = query.split('\n')
    line_scores = []

    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue

        # Get context based on type
        if context_type == 'before':
            context = get_sql_context_before(lines, line_idx, context_size)
        elif context_type == 'after':
            context = get_sql_context_after(lines, line_idx, context_size)
        else:  # full
            context = get_sql_context_full(lines, line_idx)

        # Compute embeddings
        line_emb = model.encode([line], convert_to_numpy=True)
        context_emb = model.encode([context], convert_to_numpy=True)

        # Compute cosine similarity
        similarity = cosine_similarity(line_emb, context_emb)[0][0]

        line_scores.append((line_idx, line, similarity))

    return line_scores


def predict_buggy_lines(
    line_scores: List[Tuple[int, str, float]],
    threshold: float = 0.5,
    strategy: str = 'threshold'
) -> List[int]:
    """
    Predict which lines are buggy based on similarity scores.

    Args:
        line_scores: List of (line_index, line_text, similarity_score)
        threshold: Threshold for buggy classification (lower = more likely buggy)
        strategy: 'threshold' or 'percentile'

    Returns:
        List of buggy line indices
    """
    if not line_scores:
        return []

    buggy_indices = []

    if strategy == 'threshold':
        # Lines with similarity below threshold are buggy
        for line_idx, line, score in line_scores:
            if score < threshold:
                buggy_indices.append(line_idx)

    elif strategy == 'percentile':
        # Lines in bottom percentile are buggy
        scores = [score for _, _, score in line_scores]
        percentile_threshold = np.percentile(scores, threshold * 100)

        for line_idx, line, score in line_scores:
            if score <= percentile_threshold:
                buggy_indices.append(line_idx)

    return buggy_indices


def evaluate_on_nl2sql_bugs(
    model: SentenceTransformer,
    benchmark_data: List[Dict[str, Any]],
    threshold: float = 0.5,
    context_type: str = 'full',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on NL2SQL-Bugs benchmark.

    Args:
        model: Trained model
        benchmark_data: List of benchmark entries
        threshold: Similarity threshold for buggy classification
        context_type: Context type for inference
        verbose: Print progress

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*80)
    print("EVALUATING ON NL2SQL-BUGS BENCHMARK")
    print("="*80)
    print(f"Threshold: {threshold}")
    print(f"Context type: {context_type}")

    # Prepare predictions and labels
    query_predictions = []  # Query-level: is any line buggy?
    query_labels = []       # Query-level ground truth

    # Track per-error-type performance
    error_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for entry in tqdm(benchmark_data, desc="Evaluating", disable=not verbose):
        try:
            processed = preprocess_nl2sql_entry(entry)

            query = processed['query']
            is_correct = processed['is_correct']
            error_type = processed['error_type']

            # Skip empty queries
            if not query or not query.strip():
                continue

            # Compute line similarity scores
            line_scores = compute_line_similarity_scores(
                model,
                query,
                context_type=context_type
            )

            # Predict buggy lines
            predicted_buggy_lines = predict_buggy_lines(
                line_scores,
                threshold=threshold,
                strategy='threshold'
            )

            # Query-level prediction: has buggy lines?
            has_buggy_lines = len(predicted_buggy_lines) > 0

            # Ground truth: query is incorrect (has bugs)
            has_bugs = not is_correct

            # Store predictions and labels
            query_predictions.append(int(has_buggy_lines))
            query_labels.append(int(has_bugs))

            # Track per-error-type
            if not is_correct:  # Only for incorrect queries
                error_type_stats[error_type]['total'] += 1
                if has_buggy_lines:
                    error_type_stats[error_type]['correct'] += 1

        except Exception as e:
            if verbose:
                print(f"\nWarning: Failed to process entry: {e}")
            continue

    # Compute metrics
    accuracy = accuracy_score(query_labels, query_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        query_labels,
        query_predictions,
        average='binary',
        pos_label=1  # 1 = has bugs
    )

    # Confusion matrix
    cm = confusion_matrix(query_labels, query_predictions)

    # Type-specific accuracy
    type_specific_acc = {}
    for error_type, stats in error_type_stats.items():
        if stats['total'] > 0:
            type_specific_acc[error_type] = stats['correct'] / stats['total']

    # Compile results
    results = {
        'overall_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'type_specific_accuracy': type_specific_acc,
        'num_samples': len(query_labels),
        'num_correct_predictions': sum(np.array(query_labels) == np.array(query_predictions)),
        'threshold': threshold,
        'context_type': context_type,
    }

    return results


def print_evaluation_results(results: Dict[str, Any]):
    """
    Print formatted evaluation results.

    Args:
        results: Results dictionary from evaluate_on_nl2sql_bugs
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print(f"\nDataset: {results['num_samples']} samples")
    print(f"Threshold: {results['threshold']}")
    print(f"Context type: {results['context_type']}")

    print(f"\n{'OVERALL METRICS':^80}")
    print("-"*80)
    print(f"  Overall Accuracy:        {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print(f"  Precision (buggy):       {results['precision']:.4f}")
    print(f"  Recall (buggy):          {results['recall']:.4f}")
    print(f"  F1-Score (buggy):        {results['f1_score']:.4f}")

    print(f"\n{'CONFUSION MATRIX':^80}")
    print("-"*80)
    cm = np.array(results['confusion_matrix'])
    print("                 Predicted")
    print("               Correct  Buggy")
    print(f"  Actual Correct  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"         Buggy    {cm[1,0]:5d}   {cm[1,1]:5d}")

    if results['type_specific_accuracy']:
        print(f"\n{'TYPE-SPECIFIC ACCURACY':^80}")
        print("-"*80)
        for error_type, acc in sorted(results['type_specific_accuracy'].items()):
            print(f"  {error_type:.<40} {acc:.4f} ({acc*100:.2f}%)")

    print("\n" + "="*80)


def save_results(results: Dict[str, Any], output_path: Path):
    """
    Save evaluation results to JSON file.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate SQL bug detection model on NL2SQL-Bugs benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model directory'
    )

    parser.add_argument(
        '--benchmark-data',
        type=str,
        default='data/nl2sql_bugs/NL2SQL-Bugs.json',
        help='Path to NL2SQL-Bugs.json file'
    )

    parser.add_argument(
        '--download-benchmark',
        action='store_true',
        help='Download benchmark data if not found'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Similarity threshold for buggy classification (default: 0.5)'
    )

    parser.add_argument(
        '--context-type',
        type=str,
        default='full',
        choices=['before', 'after', 'full'],
        help='Context type for inference (default: full)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/nl2sql_bugs_eval',
        help='Output directory for results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Download benchmark if requested
    benchmark_path = Path(args.benchmark_data)
    if not benchmark_path.exists() and args.download_benchmark:
        download_nl2sql_bugs(benchmark_path.parent)

    # Load benchmark data
    try:
        benchmark_data = load_nl2sql_bugs(benchmark_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo download the benchmark:")
        print(f"  python {sys.argv[0]} --download-benchmark --benchmark-data {args.benchmark_data}")
        print("\nOr manually download from:")
        print("  https://github.com/HKUSTDial/NL2SQL-Bugs-Benchmark/tree/main/data")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    try:
        model = SentenceTransformer(args.model_path, device=device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run evaluation
    results = evaluate_on_nl2sql_bugs(
        model=model,
        benchmark_data=benchmark_data,
        threshold=args.threshold,
        context_type=args.context_type,
        verbose=True
    )

    # Print results
    print_evaluation_results(results)

    # Save results
    output_path = Path(args.output_dir) / f"results_threshold_{args.threshold}.json"
    save_results(results, output_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
