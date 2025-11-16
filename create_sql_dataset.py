#!/usr/bin/env python3
"""
Create HuggingFace Dataset for SQL Bug Detection

This script creates a HuggingFace dataset from processed SQL bugs.
It implements 3x context augmentation: each SQL line generates 3 training examples
with different context windows (before/after/full).

Based on create_hf_dataset_optimized.py from the Python bug detection pipeline.

Usage:
    python create_sql_dataset.py \
        --input-json data/processed_sql_dataset.json \
        --output-dir data/sql_hf_dataset \
        --context-size 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, List
from tqdm import tqdm

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from sql_utils import create_sql_dataset_entry


def generate_sql_entries(
    processed_data: List[Dict[str, Any]],
    target_split: str,
    context_size: int = 3,
    full_context_only: bool = False
) -> Iterator[Dict[str, Any]]:
    """
    Generate dataset entries for a specific split using generators for memory efficiency.

    Based on create_hf_dataset_optimized.py:133-204.

    Args:
        processed_data: List of processed SQL entries
        target_split: 'train', 'validation', or 'test'
        context_size: Number of context lines before/after (default: 3)
        full_context_only: If True, only generate full context entries (no augmentation)

    Yields:
        Dataset entries with 3x context augmentation (or 1x if full_context_only)
    """
    for entry in processed_data:
        # Filter by split
        if entry['split'] != target_split:
            continue

        # Determine which query to use for dataset generation
        # For buggy entries, use buggy_query; for stable, use correct_query
        query = entry['buggy_query'] if entry['dataset_type'] == 'buggy' else entry['correct_query']

        # Prepare metadata
        metadata = {
            'query_type': entry.get('query_type', 'UNKNOWN'),
            'error_type': entry.get('error_type', 'unknown'),
            'user_request': entry.get('user_request', ''),
            'table_name': entry.get('table_name', ''),
            'database': entry.get('database', ''),
        }

        # Generate entries with context augmentation
        for sql_entry in create_sql_dataset_entry(
            sql_query=query,
            buggy_lines_indices=entry['buggy_lines_indices'],
            dataset_type=entry['dataset_type'],
            split=target_split,
            metadata=metadata,
            context_size=context_size,
            full_context_only=full_context_only
        ):
            yield sql_entry


def create_sql_hf_dataset(
    processed_data: List[Dict[str, Any]],
    context_size: int = 3,
    verbose: bool = True,
    full_context_only: bool = False
) -> DatasetDict:
    """
    Create HuggingFace DatasetDict with train/validation/test splits.

    Uses generators to avoid loading entire dataset into memory.

    Args:
        processed_data: List of processed SQL entries
        context_size: Number of context lines before/after
        verbose: Print progress information
        full_context_only: If True, only generate full context entries (no augmentation)

    Returns:
        DatasetDict with train/validation/test splits
    """
    # Define dataset features
    features = Features({
        'current_line': Value('string'),
        'line_index': Value('int32'),
        'context': Value('string'),
        'context_type': ClassLabel(names=['before', 'after', 'full']),
        'score': ClassLabel(names=['buggy', 'correct']),  # 0=buggy, 1=correct
        'split': ClassLabel(names=['train', 'validation', 'test']),
        'dataset_type': ClassLabel(names=['buggy', 'stable']),
        'query_type': Value('string'),
        'error_type': Value('string'),
        'user_request': Value('string'),
        'table_name': Value('string'),
        'database': Value('string'),
    })

    datasets = {}

    for split_name in ['train', 'validation', 'test']:
        if verbose:
            print(f"\nCreating {split_name} dataset...")

        # Create dataset from generator
        # Use default arguments to capture current values in closure
        dataset = Dataset.from_generator(
            lambda s=split_name, cs=context_size, fco=full_context_only: generate_sql_entries(processed_data, s, cs, fco),
            features=features
        )

        datasets[split_name] = dataset

        if verbose:
            print(f"  {split_name}: {len(dataset)} entries")

            # Print class distribution
            if len(dataset) > 0:
                score_counts = {
                    'buggy': sum(1 for item in dataset if item['score'] == 0),
                    'correct': sum(1 for item in dataset if item['score'] == 1)
                }
                print(f"    Buggy: {score_counts['buggy']}")
                print(f"    Correct: {score_counts['correct']}")

    # Create DatasetDict
    dataset_dict = DatasetDict(datasets)

    return dataset_dict


def print_dataset_statistics(dataset_dict: DatasetDict):
    """
    Print detailed statistics about the dataset.

    Args:
        dataset_dict: HuggingFace DatasetDict
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_entries = sum(len(dataset_dict[split]) for split in dataset_dict.keys())
    print(f"\nTotal entries: {total_entries:,}")

    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset_dict:
            continue

        dataset = dataset_dict[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Total: {len(dataset):,}")

        # Context type distribution
        context_counts = {}
        for item in dataset:
            ctx_type = item['context_type']
            if isinstance(ctx_type, int):
                ctx_type = dataset.features['context_type'].int2str(ctx_type)
            context_counts[ctx_type] = context_counts.get(ctx_type, 0) + 1

        print(f"  Context types:")
        for ctx_type, count in sorted(context_counts.items()):
            print(f"    {ctx_type}: {count:,}")

        # Score distribution
        score_counts = {'buggy': 0, 'correct': 0}
        for item in dataset:
            score = item['score']
            if score == 0 or score == 'buggy':
                score_counts['buggy'] += 1
            else:
                score_counts['correct'] += 1

        print(f"  Scores:")
        print(f"    Buggy (0): {score_counts['buggy']:,} ({score_counts['buggy']/len(dataset)*100:.1f}%)")
        print(f"    Correct (1): {score_counts['correct']:,} ({score_counts['correct']/len(dataset)*100:.1f}%)")

        # Dataset type distribution
        type_counts = {}
        for item in dataset:
            dtype = item['dataset_type']
            if isinstance(dtype, int):
                dtype = dataset.features['dataset_type'].int2str(dtype)
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        print(f"  Dataset types:")
        for dtype, count in sorted(type_counts.items()):
            print(f"    {dtype}: {count:,}")

    print("\n" + "=" * 60)


def print_example_entries(dataset_dict: DatasetDict, num_examples: int = 2):
    """
    Print example entries from the dataset.

    Args:
        dataset_dict: HuggingFace DatasetDict
        num_examples: Number of examples to print per split
    """
    print("\n" + "=" * 60)
    print("EXAMPLE ENTRIES")
    print("=" * 60)

    for split_name in ['train']:  # Only show train examples
        if split_name not in dataset_dict or len(dataset_dict[split_name]) == 0:
            continue

        dataset = dataset_dict[split_name]
        print(f"\n{split_name.upper()} Examples:")

        for i in range(min(num_examples, len(dataset))):
            entry = dataset[i]
            print(f"\n  Example {i+1}:")
            print(f"    Current line: {entry['current_line']}")
            print(f"    Line index: {entry['line_index']}")

            # Handle context_type (can be int or string)
            ctx_type = entry['context_type']
            if isinstance(ctx_type, int):
                ctx_type = dataset.features['context_type'].int2str(ctx_type)
            print(f"    Context type: {ctx_type}")

            # Show first 100 chars of context
            context = entry['context']
            print(f"    Context: {context[:100]}{'...' if len(context) > 100 else ''}")

            # Handle score (can be int or string)
            score = entry['score']
            if isinstance(score, int):
                score_label = dataset.features['score'].int2str(score)
            else:
                score_label = score
            print(f"    Score: {score} ({score_label})")

            print(f"    Query type: {entry['query_type']}")
            print(f"    Error type: {entry['error_type']}")

    print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset for SQL bug detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_sql_dataset.py \
    --input-json data/processed_sql_dataset.json \
    --output-dir data/sql_hf_dataset

  # With custom context size
  python create_sql_dataset.py \
    --input-json data/processed_sql_dataset.json \
    --output-dir data/sql_hf_dataset \
    --context-size 5

  # Quiet mode (no statistics)
  python create_sql_dataset.py \
    --input-json data/processed_sql_dataset.json \
    --output-dir data/sql_hf_dataset \
    --quiet
        """
    )

    parser.add_argument(
        '--input-json',
        type=str,
        required=True,
        help='Path to processed SQL dataset JSON from process_sql_bugs.py'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/sql_hf_dataset',
        help='Output directory for HuggingFace dataset (default: data/sql_hf_dataset)'
    )

    parser.add_argument(
        '--context-size',
        type=int,
        default=3,
        help='Number of context lines before/after (default: 3)'
    )

    parser.add_argument(
        '--full-context-only',
        action='store_true',
        help='Only use full SQL query as context (no 3x augmentation with before/after)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress statistics and examples output'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load processed data
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading processed data from: {input_path}")
    with open(input_path, 'r') as f:
        processed_data = json.load(f)

    if not isinstance(processed_data, list):
        print("Error: Input JSON must be a list of entries")
        sys.exit(1)

    print(f"Loaded {len(processed_data)} processed entries")

    # Show mode
    if args.full_context_only:
        print("Mode: Full context only (no 3x augmentation)")
    else:
        print(f"Mode: 3x context augmentation (context_size={args.context_size})")

    # Create HuggingFace dataset
    dataset_dict = create_sql_hf_dataset(
        processed_data,
        context_size=args.context_size,
        verbose=not args.quiet,
        full_context_only=args.full_context_only
    )

    # Save dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to: {output_dir}")
    dataset_dict.save_to_disk(str(output_dir))

    print(f"Dataset saved successfully!")

    # Print statistics and examples
    if not args.quiet:
        print_dataset_statistics(dataset_dict)
        print_example_entries(dataset_dict, num_examples=2)

    print("\nDone!")
    print(f"\nTo load the dataset:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_dir}')")


if __name__ == "__main__":
    main()
