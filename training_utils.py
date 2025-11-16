"""
Utility functions for training embedding models with different loss functions.

Provides helper functions for:
- Class weight computation
- Data preparation for different loss functions (Cosine, MNR, SupCon)
- LoRA setup
- Configuration management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers import losses as st_losses
from pytorch_metric_learning import losses as pml_losses
from peft import LoraConfig, TaskType


def compute_class_weights(dataset) -> Tuple[np.ndarray, List[float]]:
    """
    Compute class weights for weighted sampling to handle imbalance.

    Args:
        dataset: HuggingFace dataset with 'score' field (0 or 1)

    Returns:
        Tuple of (class_weights array, sample_weights list)
    """
    print("Computing class weights for balanced sampling...")

    # Count occurrences of each class
    scores = [item['score'] for item in dataset]
    class_counts = Counter(scores)

    print(f"  Class distribution:")
    print(f"    Buggy (score=0): {class_counts[0]:,} samples")
    print(f"    Correct (score=1): {class_counts[1]:,} samples")

    # Compute weights: inverse of class frequency
    total_samples = len(scores)
    class_weights = np.array([
        total_samples / (len(class_counts) * class_counts[0]),  # weight for class 0
        total_samples / (len(class_counts) * class_counts[1])   # weight for class 1
    ])

    print(f"  Class weights: buggy={class_weights[0]:.4f}, correct={class_weights[1]:.4f}")

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[score] for score in scores]

    return class_weights, sample_weights


def create_cosine_examples(dataset, max_samples: int = None) -> List[InputExample]:
    """
    Create InputExample format for CosineSimilarityLoss.

    Uses ALL data (both buggy and correct lines).
    Format: InputExample(texts=[current_line, context], label=score)

    Args:
        dataset: HuggingFace dataset with current_line, context, and score
        max_samples: Optional limit on number of samples

    Returns:
        List of InputExample objects
    """
    print(f"Creating Cosine Similarity examples...")
    examples = []

    dataset_slice = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    for idx, item in enumerate(dataset_slice):
        if idx % 50000 == 0:
            print(f"  Processed {idx:,}/{len(dataset_slice):,} examples")

        example = InputExample(
            texts=[item['current_line'], item['context']],
            label=float(item['score'])  # 0.0 or 1.0
        )
        examples.append(example)

    print(f"✓ Created {len(examples):,} Cosine examples")
    return examples


def create_mnr_examples(dataset, max_samples: int = None) -> List[InputExample]:
    """
    Create InputExample format for MultipleNegativesRankingLoss.

    Uses ONLY positive pairs (score=1) - correct lines only.
    Format: InputExample(texts=[current_line, context])
    No label needed - all are treated as positive pairs.

    Args:
        dataset: HuggingFace dataset with current_line, context, and score
        max_samples: Optional limit on number of samples

    Returns:
        List of InputExample objects (only positive pairs)
    """
    print(f"Creating MNR (Multiple Negatives Ranking) examples...")
    print(f"  Note: Using only POSITIVE pairs (score=1)")
    examples = []

    dataset_slice = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    positive_count = 0
    for idx, item in enumerate(dataset_slice):
        if idx % 50000 == 0:
            print(f"  Processed {idx:,}/{len(dataset_slice):,} examples")

        # Only include positive pairs (correct lines)
        if item['score'] == 1:
            example = InputExample(
                texts=[item['current_line'], item['context']]
                # No label - MNR treats all as positive pairs
            )
            examples.append(example)
            positive_count += 1

    print(f"✓ Created {len(examples):,} MNR examples (positive pairs only)")
    print(f"  Skipped {dataset_slice.num_rows - positive_count:,} negative samples")

    return examples


def create_supcon_data(dataset, max_samples: int = None) -> Tuple[List[str], List[str], List[int]]:
    """
    Create data format for Supervised Contrastive Loss.

    Uses ALL data (both buggy and correct lines).
    Returns separate lists for lines, contexts, and labels.

    Args:
        dataset: HuggingFace dataset with current_line, context, and score
        max_samples: Optional limit on number of samples

    Returns:
        Tuple of (lines_list, contexts_list, labels_list)
    """
    print(f"Creating SupCon (Supervised Contrastive) data...")
    lines = []
    contexts = []
    labels = []

    dataset_slice = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    for idx, item in enumerate(dataset_slice):
        if idx % 50000 == 0:
            print(f"  Processed {idx:,}/{len(dataset_slice):,} examples")

        lines.append(item['current_line'])
        contexts.append(item['context'])
        labels.append(item['score'])  # 0 or 1

    print(f"✓ Created {len(lines):,} SupCon samples")

    # Print class distribution
    label_counts = Counter(labels)
    print(f"  Buggy (label=0): {label_counts[0]:,}")
    print(f"  Correct (label=1): {label_counts[1]:,}")

    return lines, contexts, labels


def setup_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: List[str] = None
) -> LoraConfig:
    """
    Create LoRA configuration for parameter-efficient fine-tuning.

    Args:
        rank: LoRA rank (default: 16)
        alpha: LoRA alpha scaling factor (default: 32, typically 2*rank)
        dropout: LoRA dropout (default: 0.1)
        target_modules: Target modules for LoRA (default: Q, K, V projections)

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj']

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
    )

    return lora_config


def get_output_dir(base_dir: str, model_name: str, loss_function: str) -> Path:
    """
    Generate output directory path based on model and loss function.

    Args:
        base_dir: Base directory for models
        model_name: Name of the model being fine-tuned
        loss_function: Loss function being used (cosine, mnr, supcon)

    Returns:
        Path object for output directory
    """
    # Extract model name (last part after /)
    model_short_name = model_name.split('/')[-1]

    # Create descriptive directory name
    dir_name = f"finetuned-{model_short_name}-{loss_function}"

    output_path = Path(base_dir) / dir_name

    return output_path


def print_config(config_dict: Dict[str, Any], title: str = "Configuration"):
    """
    Pretty print configuration dictionary.

    Args:
        config_dict: Dictionary of configuration parameters
        title: Title for the configuration section
    """
    print("=" * 80)
    print(title)
    print("=" * 80)

    for key, value in config_dict.items():
        # Format key nicely (replace underscores with spaces, capitalize)
        formatted_key = key.replace('_', ' ').title()

        # Format value based on type
        if isinstance(value, Path):
            formatted_value = str(value)
        elif isinstance(value, bool):
            formatted_value = "Yes" if value else "No"
        elif isinstance(value, (int, float)):
            if isinstance(value, int) and value >= 1000:
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
        else:
            formatted_value = str(value)

        print(f"  {formatted_key:.<40} {formatted_value}")

    print("=" * 80)


def print_training_summary(
    loss_function: str,
    total_steps: int,
    steps_per_epoch: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    use_lora: bool,
    use_amp: bool,
    use_weighted_sampling: bool
):
    """
    Print training summary before starting training.

    Args:
        loss_function: Name of loss function
        total_steps: Total training steps
        steps_per_epoch: Steps per epoch
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_lora: Whether using LoRA
        use_amp: Whether using AMP
        use_weighted_sampling: Whether using weighted sampling
    """
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    print(f"\nLoss Function: {loss_function.upper()}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    print(f"\nOptimizations:")
    print(f"  LoRA: {'Yes' if use_lora else 'No'}")
    print(f"  AMP (Mixed Precision): {'Yes' if use_amp else 'No'}")
    print(f"  Weighted Sampling: {'Yes' if use_weighted_sampling else 'No'}")

    print("\n" + "=" * 80)


class WeightedEnsembleLoss(nn.Module):
    """
    Weighted ensemble of multiple loss functions.

    Combines CosineSimilarityLoss, MultipleNegativesRankingLoss, and/or
    SupervisedContrastiveLoss with configurable weights.

    Each loss function has different input requirements:
    - CosineSimilarityLoss: requires labels
    - MultipleNegativesRankingLoss: no labels, only positive pairs
    - SupConLoss: requires embeddings and labels

    Args:
        model: SentenceTransformer model
        loss_types: List of loss types to combine ['cosine', 'mnr', 'supcon']
        weights: Optional list of weights for each loss (default: equal weights)
        temperature: Temperature for SupCon loss (default: 0.07)
    """

    def __init__(
        self,
        model: SentenceTransformer,
        loss_types: List[str],
        weights: Optional[List[float]] = None,
        temperature: float = 0.07
    ):
        super().__init__()
        self.model = model
        self.loss_types = loss_types

        # Set weights (default to equal weights)
        if weights is None:
            self.weights = [1.0 / len(loss_types)] * len(loss_types)
        else:
            assert len(weights) == len(loss_types), "Number of weights must match number of loss types"
            # Normalize weights to sum to 1.0
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # Initialize loss functions
        self.losses = {}
        for loss_type in loss_types:
            if loss_type == 'cosine':
                self.losses['cosine'] = st_losses.CosineSimilarityLoss(model)
            elif loss_type == 'mnr':
                self.losses['mnr'] = st_losses.MultipleNegativesRankingLoss(model)
            elif loss_type == 'supcon':
                self.losses['supcon'] = pml_losses.SupConLoss(temperature=temperature)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        print(f"✓ WeightedEnsembleLoss initialized:")
        for loss_type, weight in zip(loss_types, self.weights):
            print(f"  - {loss_type.upper()}: weight={weight:.3f}")

    def forward(
        self,
        sentence_features: List[Dict[str, torch.Tensor]],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted ensemble loss.

        Args:
            sentence_features: List of dicts containing 'input_ids', 'attention_mask', etc.
                              For line-context pairs: [line_features, context_features]
            labels: Binary labels (0=buggy, 1=correct)

        Returns:
            Combined weighted loss
        """
        total_loss = 0.0
        individual_losses = {}

        # Get embeddings once (reuse for all losses)
        # This is more efficient than having each loss compute separately
        # Note: We need gradients enabled for backpropagation
        embeddings = []
        for features in sentence_features:
            emb = self.model(features)['sentence_embedding']
            embeddings.append(emb)

        # Compute each loss
        for loss_type, weight in zip(self.loss_types, self.weights):
            if loss_type == 'cosine':
                # CosineSimilarityLoss expects sentence_features and labels
                loss_value = self.losses['cosine'](sentence_features, labels)
                individual_losses['cosine'] = loss_value.item()
                total_loss += weight * loss_value

            elif loss_type == 'mnr':
                # MultipleNegativesRankingLoss only uses positive pairs
                # Filter to only positive samples (label=1)
                positive_mask = (labels == 1).cpu().numpy()

                if positive_mask.sum() > 0:
                    # Filter sentence_features to only positive pairs
                    filtered_features = []
                    for features in sentence_features:
                        filtered_dict = {}
                        for key, value in features.items():
                            if isinstance(value, torch.Tensor):
                                filtered_dict[key] = value[positive_mask]
                            else:
                                filtered_dict[key] = value
                        filtered_features.append(filtered_dict)

                    loss_value = self.losses['mnr'](filtered_features, None)
                    individual_losses['mnr'] = loss_value.item()
                    total_loss += weight * loss_value
                else:
                    # No positive samples in batch, skip MNR
                    individual_losses['mnr'] = 0.0

            elif loss_type == 'supcon':
                # SupConLoss expects normalized embeddings and labels
                # Combine line and context embeddings via averaging
                combined_embeddings = (embeddings[0] + embeddings[1]) / 2
                combined_embeddings = F.normalize(combined_embeddings, p=2, dim=1)

                loss_value = self.losses['supcon'](combined_embeddings, labels)
                individual_losses['supcon'] = loss_value.item()
                total_loss += weight * loss_value

        # Store individual losses for logging
        self.last_individual_losses = individual_losses

        return total_loss

    def get_config_dict(self) -> Dict[str, Any]:
        """Return configuration dictionary for saving."""
        return {
            'loss_types': self.loss_types,
            'weights': self.weights,
            'temperature': self.losses.get('supcon', None).temperature if 'supcon' in self.losses else None
        }
