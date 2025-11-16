"""
SQL Bug Detection Utility Functions

This module provides utility functions for SQL bug detection pipeline,
adapted from the Python bug detection utilities.

Functions:
    - infer_buggy_sql_lines_from_diff: Find buggy lines using difflib
    - get_sql_context_before: Get context with preceding lines
    - get_sql_context_after: Get context with following lines
    - get_sql_context_full: Get full query context
    - create_sql_dataset_entry: Generate dataset entries with 3x context
    - compute_sql_class_weights: Compute class weights for balanced training
    - normalize_sql_query: Normalize SQL query for comparison
"""

import difflib
import re
from typing import List, Dict, Any, Iterator, Optional
from collections import Counter


def normalize_sql_query(query: str) -> str:
    """
    Normalize SQL query for consistent comparison.

    Args:
        query: SQL query string

    Returns:
        Normalized query string
    """
    # Remove leading/trailing whitespace
    query = query.strip()

    # Remove empty lines
    lines = [line for line in query.split('\n') if line.strip()]

    return '\n'.join(lines)


def infer_buggy_sql_lines_from_diff(
    buggy_query: str,
    correct_query: str,
    normalize: bool = True
) -> List[int]:
    """
    Infer which lines are buggy by comparing buggy and correct SQL queries.

    This function uses difflib to compute unified diff between two SQL queries
    and identifies which lines differ. Based on process_pytracebugs.py:134-169.

    Args:
        buggy_query: The buggy SQL query
        correct_query: The correct SQL query
        normalize: Whether to normalize queries before comparison

    Returns:
        List of 0-based line indices that are buggy

    Example:
        >>> buggy = "SELECT * FROM users\\nWHERE age >= 18"
        >>> correct = "SELECT * FROM users\\nWHERE age > 18"
        >>> infer_buggy_sql_lines_from_diff(buggy, correct)
        [1]  # Line 1 (WHERE clause) is buggy
    """
    if normalize:
        buggy_query = normalize_sql_query(buggy_query)
        correct_query = normalize_sql_query(correct_query)

    buggy_lines = buggy_query.split('\n')
    correct_lines = correct_query.split('\n')

    # Compute unified diff
    diff = list(difflib.unified_diff(
        buggy_lines,
        correct_lines,
        lineterm='',
        n=0  # No context lines
    ))

    buggy_indices = []
    current_line = 0

    for line in diff:
        # Parse hunk header: @@ -start,count +start,count @@
        if line.startswith('@@'):
            # Extract starting line number for buggy version (after -)
            match = re.search(r'-(\d+)(?:,(\d+))?', line)
            if match:
                start_line = int(match.group(1))
                current_line = start_line - 1  # Convert to 0-based

        # Lines starting with '-' are in buggy version but not in correct
        elif line.startswith('-') and not line.startswith('---'):
            buggy_indices.append(current_line)
            current_line += 1

        # Lines starting with ' ' are in both versions
        elif line.startswith(' '):
            current_line += 1

        # Lines starting with '+' are only in correct version (skip)
        # Don't increment current_line for '+' lines

    # Remove duplicates and sort
    return sorted(list(set(buggy_indices)))


def get_sql_context_before(
    lines: List[str],
    line_idx: int,
    context_size: int = 3
) -> str:
    """
    Get SQL context with current line and preceding lines.

    Based on create_hf_dataset_optimized.py:22-36.

    Args:
        lines: List of SQL query lines
        line_idx: Index of current line (0-based)
        context_size: Number of preceding lines to include (default: 3)

    Returns:
        Context string with current line and up to 3 preceding lines

    Example:
        >>> lines = ["SELECT *", "FROM users", "WHERE age > 18", "ORDER BY name"]
        >>> get_sql_context_before(lines, 2, context_size=3)
        "SELECT *\\nFROM users\\nWHERE age > 18"
    """
    start_idx = max(0, line_idx - context_size)
    context_lines = lines[start_idx : line_idx + 1]
    return '\n'.join(context_lines)


def get_sql_context_after(
    lines: List[str],
    line_idx: int,
    context_size: int = 3
) -> str:
    """
    Get SQL context with current line and following lines.

    Based on create_hf_dataset_optimized.py:39-53.

    Args:
        lines: List of SQL query lines
        line_idx: Index of current line (0-based)
        context_size: Number of following lines to include (default: 3)

    Returns:
        Context string with current line and up to 3 following lines

    Example:
        >>> lines = ["SELECT *", "FROM users", "WHERE age > 18", "ORDER BY name"]
        >>> get_sql_context_after(lines, 1, context_size=3)
        "FROM users\\nWHERE age > 18\\nORDER BY name"
    """
    end_idx = min(len(lines), line_idx + context_size + 1)
    context_lines = lines[line_idx : end_idx]
    return '\n'.join(context_lines)


def get_sql_context_full(
    lines: List[str],
    line_idx: int
) -> str:
    """
    Get full SQL query as context.

    Based on create_hf_dataset_optimized.py:56-67.

    Args:
        lines: List of SQL query lines
        line_idx: Index of current line (0-based, not used but kept for API consistency)

    Returns:
        Full SQL query as context string

    Example:
        >>> lines = ["SELECT *", "FROM users", "WHERE age > 18"]
        >>> get_sql_context_full(lines, 1)
        "SELECT *\\nFROM users\\nWHERE age > 18"
    """
    return '\n'.join(lines)


def create_sql_dataset_entry(
    sql_query: str,
    buggy_lines_indices: List[int],
    dataset_type: str,
    split: str,
    metadata: Optional[Dict[str, Any]] = None,
    context_size: int = 3,
    full_context_only: bool = False
) -> Iterator[Dict[str, Any]]:
    """
    Generate dataset entries for a SQL query with 3x context augmentation.

    Each SQL line generates 3 entries (before/after/full context) unless
    full_context_only is True, in which case only full context is generated.
    Based on create_hf_dataset_optimized.py:70-131.

    Args:
        sql_query: SQL query string
        buggy_lines_indices: List of 0-based indices of buggy lines
        dataset_type: 'buggy' or 'stable'
        split: 'train', 'validation', or 'test'
        metadata: Optional metadata dict (query_type, error_type, etc.)
        context_size: Number of context lines before/after (default: 3)
        full_context_only: If True, only generate full context entries (no augmentation)

    Yields:
        Dictionary entries for HuggingFace dataset

    Example:
        >>> query = "SELECT *\\nFROM users\\nWHERE age >= 18"
        >>> entries = list(create_sql_dataset_entry(query, [2], 'buggy', 'train'))
        >>> len(entries)  # 3 lines * 3 contexts = 9 entries
        9
        >>> entries = list(create_sql_dataset_entry(query, [2], 'buggy', 'train', full_context_only=True))
        >>> len(entries)  # 3 lines * 1 context = 3 entries
        3
    """
    if metadata is None:
        metadata = {}

    lines = sql_query.split('\n')

    # Define context functions based on mode
    if full_context_only:
        # Only use full context (no augmentation)
        context_functions = {
            'full': lambda idx: get_sql_context_full(lines, idx),
        }
    else:
        # 3x augmentation with before/after/full contexts
        context_functions = {
            'before': lambda idx: get_sql_context_before(lines, idx, context_size),
            'after': lambda idx: get_sql_context_after(lines, idx, context_size),
            'full': lambda idx: get_sql_context_full(lines, idx),
        }

    for line_idx, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            continue

        # Determine if line is buggy
        is_buggy = line_idx in buggy_lines_indices
        score = 0 if is_buggy else 1

        # Generate entry for each context type
        for context_type, context_fn in context_functions.items():
            entry = {
                'current_line': line,
                'line_index': line_idx,
                'context': context_fn(line_idx),
                'context_type': context_type,
                'score': score,
                'split': split,
                'dataset_type': dataset_type,
            }

            # Add metadata fields
            entry.update(metadata)

            yield entry


def compute_sql_class_weights(
    dataset,
    score_column: str = 'score'
) -> Dict[int, float]:
    """
    Compute class weights for balanced training.

    Based on training_utils.py:24-56.

    Args:
        dataset: HuggingFace dataset or list of dicts with score field
        score_column: Name of the score column (default: 'score')

    Returns:
        Dictionary mapping class (0 or 1) to weight

    Example:
        >>> dataset = [{'score': 0}, {'score': 1}, {'score': 1}, {'score': 1}]
        >>> weights = compute_sql_class_weights(dataset)
        >>> weights[0] > weights[1]  # Buggy class (0) gets higher weight
        True
    """
    # Extract scores
    if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
        # HuggingFace dataset or list
        scores = [item[score_column] for item in dataset]
    else:
        raise ValueError("Dataset must support indexing and len()")

    # Count classes
    class_counts = Counter(scores)
    total = len(scores)

    # Compute inverse frequency weights
    class_weights = {}
    num_classes = len(class_counts)

    for class_label, count in class_counts.items():
        # Inverse frequency: total / (num_classes * count)
        class_weights[class_label] = total / (num_classes * count)

    return class_weights


def compute_sample_weights(
    dataset,
    class_weights: Dict[int, float],
    score_column: str = 'score'
) -> List[float]:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Args:
        dataset: HuggingFace dataset or list of dicts
        class_weights: Dictionary mapping class to weight
        score_column: Name of the score column

    Returns:
        List of sample weights
    """
    scores = [item[score_column] for item in dataset]
    sample_weights = [class_weights[score] for score in scores]
    return sample_weights


def extract_buggy_lines_text(
    sql_query: str,
    buggy_lines_indices: List[int]
) -> List[str]:
    """
    Extract the text of buggy lines from a SQL query.

    Based on process_pytracebugs.py:58-79.

    Args:
        sql_query: SQL query string
        buggy_lines_indices: List of 0-based indices of buggy lines

    Returns:
        List of buggy line texts

    Example:
        >>> query = "SELECT *\\nFROM users\\nWHERE age >= 18"
        >>> extract_buggy_lines_text(query, [2])
        ['WHERE age >= 18']
    """
    lines = sql_query.split('\n')
    buggy_lines = []

    for idx in buggy_lines_indices:
        if 0 <= idx < len(lines):
            buggy_lines.append(lines[idx])

    return buggy_lines


def split_dataset_by_ratio(
    total_entries: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, range]:
    """
    Split dataset indices by ratio.

    Args:
        total_entries: Total number of entries
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)

    Returns:
        Dictionary with 'train', 'validation', 'test' ranges
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    train_end = int(total_entries * train_ratio)
    val_end = train_end + int(total_entries * val_ratio)

    return {
        'train': range(0, train_end),
        'validation': range(train_end, val_end),
        'test': range(val_end, total_entries)
    }


def get_sql_query_type(sql_query: str) -> str:
    """
    Infer SQL query type from query string.

    Args:
        sql_query: SQL query string

    Returns:
        Query type (SELECT, INSERT, UPDATE, DELETE, etc.)
    """
    query_upper = sql_query.strip().upper()

    keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']

    for keyword in keywords:
        if query_upper.startswith(keyword):
            return keyword

    return 'UNKNOWN'


# Example usage and testing
if __name__ == "__main__":
    # Example: Diff computation
    buggy = """SELECT * FROM users
WHERE age >= 18
ORDER BY name"""

    correct = """SELECT * FROM users
WHERE age > 18
ORDER BY name"""

    buggy_lines = infer_buggy_sql_lines_from_diff(buggy, correct)
    print(f"Buggy lines: {buggy_lines}")  # [1]

    # Example: Context extraction
    lines = buggy.split('\n')
    print(f"\nContext before line 1: {get_sql_context_before(lines, 1)}")
    print(f"Context after line 1: {get_sql_context_after(lines, 1)}")
    print(f"Full context: {get_sql_context_full(lines, 1)}")

    # Example: Dataset entry creation
    entries = list(create_sql_dataset_entry(
        buggy,
        buggy_lines,
        'buggy',
        'train',
        {'query_type': 'SELECT', 'error_type': 'logical'}
    ))
    print(f"\nGenerated {len(entries)} dataset entries")
    print(f"First entry: {entries[0]}")
