"""
Comprehensive evaluation script for fine-tuned embedding model.

Evaluates the model on the test set with:
- Classification metrics (Accuracy, Precision, Recall, F1)
- Ranking metrics (ROC-AUC, PR-AUC, Correlation)
- Threshold optimization
- Context type analysis
- Complete visualizations
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

# ML libraries
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, classification_report
)
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def load_model_and_data(model_path: Path, data_path: Path) -> Tuple:
    """
    Load fine-tuned model and test dataset.

    Args:
        model_path: Path to fine-tuned model
        data_path: Path to HF dataset

    Returns:
        Tuple of (model, test_dataset)
    """
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = SentenceTransformer(str(model_path))
    print(f"✓ Model loaded")

    print(f"\nLoading dataset from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    dataset = load_from_disk(str(data_path))
    test_dataset = dataset['test']
    print(f"✓ Test dataset loaded: {len(test_dataset):,} samples")

    return model, test_dataset


def compute_embeddings(
    model: SentenceTransformer,
    test_dataset,
    batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Compute embeddings and cosine similarities for test set.

    Args:
        model: Fine-tuned sentence-transformers model
        test_dataset: HF test dataset
        batch_size: Batch size for encoding

    Returns:
        Tuple of (similarities, true_labels, predictions, context_types)
    """
    print(f"\nComputing embeddings...")
    print(f"  Batch size: {batch_size}")

    current_lines = []
    contexts = []
    true_labels = []
    context_types = []

    for item in test_dataset:
        current_lines.append(item['current_line'])
        contexts.append(item['context'])
        true_labels.append(item['score'])
        context_types.append(item['context_type'])

    print(f"  Encoding {len(current_lines):,} line pairs...")

    # Encode current lines
    print("  - Encoding current lines...")
    line_embeddings = model.encode(
        current_lines,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Encode contexts
    print("  - Encoding contexts...")
    context_embeddings = model.encode(
        contexts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Compute cosine similarities
    print("  - Computing cosine similarities...")
    similarities = np.array([
        cosine_similarity([line_embeddings[i]], [context_embeddings[i]])[0][0]
        for i in tqdm(range(len(line_embeddings)), desc="Similarity")
    ])

    true_labels = np.array(true_labels)

    print(f"✓ Embeddings computed")
    print(f"  Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

    return similarities, true_labels, context_types


def compute_classification_metrics(
    similarities: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Compute classification metrics at a given threshold.

    Args:
        similarities: Predicted cosine similarities
        true_labels: True labels (0 or 1)
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    # Convert similarities to binary predictions
    predictions = (similarities >= threshold).astype(int)

    # Compute metrics
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'threshold': float(threshold)
    }

    return metrics


def compute_ranking_metrics(
    similarities: np.ndarray,
    true_labels: np.ndarray
) -> Dict:
    """
    Compute ranking metrics (ROC-AUC, PR-AUC, Correlation).

    Args:
        similarities: Predicted cosine similarities
        true_labels: True labels (0 or 1)

    Returns:
        Dictionary of ranking metrics
    """
    # ROC-AUC
    roc_auc = roc_auc_score(true_labels, similarities)

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(true_labels, similarities)
    pr_auc = auc(recall, precision)

    # Correlation
    spearman_corr, spearman_pval = spearmanr(similarities, true_labels)
    pearson_corr, pearson_pval = pearsonr(similarities, true_labels)

    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'spearman_correlation': float(spearman_corr),
        'spearman_pvalue': float(spearman_pval),
        'pearson_correlation': float(pearson_corr),
        'pearson_pvalue': float(pearson_pval)
    }

    return metrics


def find_optimal_threshold(
    similarities: np.ndarray,
    true_labels: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold for a given metric.

    Args:
        similarities: Predicted cosine similarities
        true_labels: True labels
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    scores = []

    for thresh in thresholds:
        predictions = (similarities >= thresh).astype(int)

        if metric == 'f1':
            score = f1_score(true_labels, predictions, zero_division=0)
        elif metric == 'precision':
            score = precision_score(true_labels, predictions, zero_division=0)
        elif metric == 'recall':
            score = recall_score(true_labels, predictions, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return float(best_threshold), float(best_score), thresholds, scores


def evaluate_by_context_type(
    similarities: np.ndarray,
    true_labels: np.ndarray,
    context_types: List[str],
    threshold: float = 0.5
) -> Dict:
    """
    Evaluate metrics separately for each context type.

    Args:
        similarities: Predicted cosine similarities
        true_labels: True labels
        context_types: List of context types
        threshold: Decision threshold

    Returns:
        Dictionary with metrics per context type
    """
    print("\nAnalyzing by context type...")

    context_types_arr = np.array(context_types)
    unique_types = np.unique(context_types_arr)

    results = {}

    for ctx_type in unique_types:
        mask = context_types_arr == ctx_type
        ctx_similarities = similarities[mask]
        ctx_labels = true_labels[mask]

        print(f"  - {ctx_type}: {len(ctx_similarities):,} samples")

        # Classification metrics
        ctx_class_metrics = compute_classification_metrics(
            ctx_similarities, ctx_labels, threshold
        )

        # Ranking metrics
        ctx_rank_metrics = compute_ranking_metrics(ctx_similarities, ctx_labels)

        results[ctx_type] = {
            **ctx_class_metrics,
            **ctx_rank_metrics
        }

    return results


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Buggy (0)', 'Correct (1)'],
        yticklabels=['Buggy (0)', 'Correct (1)'],
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_roc_curve(
    similarities: np.ndarray,
    true_labels: np.ndarray,
    roc_auc: float,
    output_dir: Path
):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(true_labels, similarities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'roc_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_pr_curve(
    similarities: np.ndarray,
    true_labels: np.ndarray,
    pr_auc: float,
    output_dir: Path
):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(true_labels, similarities)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'precision_recall_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_similarity_distribution(
    similarities: np.ndarray,
    true_labels: np.ndarray,
    output_dir: Path
):
    """Plot distribution of similarities for buggy vs correct lines."""
    plt.figure(figsize=(10, 6))

    buggy_sim = similarities[true_labels == 0]
    correct_sim = similarities[true_labels == 1]

    plt.hist(buggy_sim, bins=50, alpha=0.6, label='Buggy (score=0)', color='red')
    plt.hist(correct_sim, bins=50, alpha=0.6, label='Correct (score=1)', color='green')

    plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')

    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Similarity Distribution: Buggy vs Correct Lines',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'similarity_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_threshold_optimization(
    thresholds: np.ndarray,
    f1_scores: np.ndarray,
    precision_scores: np.ndarray,
    recall_scores: np.ndarray,
    best_f1_thresh: float,
    output_dir: Path
):
    """Plot metrics vs threshold."""
    plt.figure(figsize=(10, 6))

    plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    plt.plot(thresholds, precision_scores, label='Precision', linewidth=2)
    plt.plot(thresholds, recall_scores, label='Recall', linewidth=2)

    plt.axvline(best_f1_thresh, color='red', linestyle='--',
                linewidth=2, label=f'Best F1 threshold={best_f1_thresh:.2f}')
    plt.axvline(0.5, color='gray', linestyle=':', linewidth=1,
                label='Default threshold=0.5')

    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Decision Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'threshold_optimization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_context_comparison(context_results: Dict, output_dir: Path):
    """Plot comparison of metrics across context types."""
    context_types = list(context_results.keys())
    metrics_to_plot = ['f1_score', 'precision', 'recall', 'roc_auc']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        values = [context_results[ct][metric] for ct in context_types]

        ax = axes[idx]
        bars = ax.bar(context_types, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} by Context Type',
                     fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'context_type_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def save_results(
    overall_metrics: Dict,
    context_results: Dict,
    optimal_thresholds: Dict,
    output_dir: Path
):
    """Save results to JSON and text report."""

    # Save JSON
    results_dict = {
        'overall_metrics': overall_metrics,
        'context_type_results': context_results,
        'optimal_thresholds': optimal_thresholds
    }

    json_path = output_dir / 'metrics_summary.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  ✓ Saved: {json_path}")

    # Save text report
    txt_path = output_dir / 'evaluation_report.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL METRICS (threshold=0.5):\n")
        f.write("-" * 80 + "\n")
        for key, value in overall_metrics.items():
            if key != 'confusion_matrix':
                f.write(f"  {key:25s}: {value:.4f}\n")

        f.write("\n\nOPTIMAL THRESHOLDS:\n")
        f.write("-" * 80 + "\n")
        for metric, (thresh, score) in optimal_thresholds.items():
            f.write(f"  Best {metric:12s}: {score:.4f} at threshold={thresh:.2f}\n")

        f.write("\n\nCONTEXT TYPE ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        for ctx_type, metrics in context_results.items():
            f.write(f"\n{ctx_type.upper()}:\n")
            for key, value in metrics.items():
                if key not in ['confusion_matrix', 'threshold']:
                    f.write(f"  {key:25s}: {value:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"  ✓ Saved: {txt_path}")


def main():
    """Main evaluation function."""

    # Configuration
    MODEL_PATH = Path('./models/finetuned-nomic-embed')
    DATA_PATH = Path('./data/hf_dataset')
    OUTPUT_DIR = Path('./evaluation_results')

    BATCH_SIZE = 128
    DEFAULT_THRESHOLD = 0.5

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {OUTPUT_DIR}")

    # Load model and data
    model, test_dataset = load_model_and_data(MODEL_PATH, DATA_PATH)

    # Compute embeddings and similarities
    similarities, true_labels, context_types = compute_embeddings(
        model, test_dataset, BATCH_SIZE
    )

    # ==================== Overall Metrics ====================
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    # Classification metrics at default threshold
    print(f"\nClassification metrics (threshold={DEFAULT_THRESHOLD}):")
    class_metrics = compute_classification_metrics(
        similarities, true_labels, DEFAULT_THRESHOLD
    )
    for key, value in class_metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key:15s}: {value:.4f}")

    # Ranking metrics
    print(f"\nRanking metrics:")
    rank_metrics = compute_ranking_metrics(similarities, true_labels)
    for key, value in rank_metrics.items():
        print(f"  {key:25s}: {value:.4f}")

    overall_metrics = {**class_metrics, **rank_metrics}

    # ==================== Threshold Optimization ====================
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)

    optimal_thresholds = {}

    for metric in ['f1', 'precision', 'recall']:
        thresh, score, all_thresholds, all_scores = find_optimal_threshold(
            similarities, true_labels, metric
        )
        optimal_thresholds[metric] = (thresh, score)
        print(f"  Best {metric:10s}: {score:.4f} at threshold={thresh:.2f}")

        # Store scores for plotting
        if metric == 'f1':
            f1_thresholds = all_thresholds
            f1_scores = all_scores
        elif metric == 'precision':
            precision_scores = all_scores
        elif metric == 'recall':
            recall_scores = all_scores

    # ==================== Context Type Analysis ====================
    print("\n" + "=" * 80)
    print("CONTEXT TYPE ANALYSIS")
    print("=" * 80)

    context_results = evaluate_by_context_type(
        similarities, true_labels, context_types, DEFAULT_THRESHOLD
    )

    for ctx_type, metrics in context_results.items():
        print(f"\n{ctx_type.upper()}:")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

    # ==================== Generate Visualizations ====================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_confusion_matrix(
        np.array(class_metrics['confusion_matrix']), OUTPUT_DIR
    )

    plot_roc_curve(
        similarities, true_labels, rank_metrics['roc_auc'], OUTPUT_DIR
    )

    plot_pr_curve(
        similarities, true_labels, rank_metrics['pr_auc'], OUTPUT_DIR
    )

    plot_similarity_distribution(similarities, true_labels, OUTPUT_DIR)

    plot_threshold_optimization(
        f1_thresholds, f1_scores, precision_scores, recall_scores,
        optimal_thresholds['f1'][0], OUTPUT_DIR
    )

    plot_context_comparison(context_results, OUTPUT_DIR)

    # ==================== Save Results ====================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    save_results(overall_metrics, context_results, optimal_thresholds, OUTPUT_DIR)

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"\nKey Findings:")
    print(f"  Overall F1-Score:  {class_metrics['f1_score']:.4f}")
    print(f"  Overall ROC-AUC:   {rank_metrics['roc_auc']:.4f}")
    print(f"  Optimal F1:        {optimal_thresholds['f1'][1]:.4f} "
          f"(threshold={optimal_thresholds['f1'][0]:.2f})")

    # Find best context type
    best_context = max(context_results.items(),
                       key=lambda x: x[1]['f1_score'])
    print(f"  Best context type: {best_context[0]} "
          f"(F1={best_context[1]['f1_score']:.4f})")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
