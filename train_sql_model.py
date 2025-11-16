#!/usr/bin/env python3
"""
Train SQL Bug Detection Model

This script trains an embedding model to detect buggy SQL lines using similarity-based
approaches. Supports multiple loss functions with selectable training strategies.

Based on train_with_contrastive.py and training_utils.py from the Python bug detection pipeline.

Loss Functions:
    - mnr: MultipleNegativesRankingLoss (recommended, state-of-the-art)
    - cosine: CosineSimilarityLoss (uses all data with labels)
    - supcon: SupervisedContrastiveLoss (contrastive learning)
    - ensemble: Weighted combination of all three losses

Usage:
    # Train with MNR loss (recommended)
    python train_sql_model.py \
        --data-dir data/sql_hf_dataset \
        --model-name nomic-ai/nomic-embed-code \
        --loss-function mnr \
        --output-dir models/sql-bug-detector \
        --epochs 3 \
        --batch-size 32

    # Train with ensemble loss
    python train_sql_model.py \
        --data-dir data/sql_hf_dataset \
        --model-name nomic-ai/nomic-embed-code \
        --loss-function ensemble \
        --ensemble-weights 0.4 0.3 0.3 \
        --use-lora \
        --epochs 3
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, IterableDataset
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses as st_losses, evaluation
from pytorch_metric_learning import losses as pml_losses
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# Import utility functions from training_utils
from training_utils import (
    compute_class_weights,
    setup_lora_config,
    get_output_dir,
    print_config,
    print_training_summary,
    WeightedEnsembleLoss,
)


@dataclass
class TrainingConfig:
    """Configuration for SQL bug detection training."""
    data_dir: str
    output_dir: str
    model_name: str
    loss_function: str

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    eval_steps: int = 2000
    max_seq_length: int = 512

    # Data sampling
    max_train_samples: Optional[int] = None
    use_weighted_sampling: bool = False

    # Optimization
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Ensemble loss weights (if loss_function='ensemble')
    ensemble_weights: List[float] = None
    ensemble_temperature: float = 0.07

    # Device
    device: Optional[str] = None

    # Misc
    seed: int = 42


class LazyInputExampleDataset(IterableDataset):
    """
    Lazy-loading dataset that generates InputExample objects on-the-fly.

    Based on train_distilroberta_full.py:30-109.
    Avoids loading all examples into memory at once.
    """

    def __init__(
        self,
        dataset: Dataset,
        loss_function: str,
        max_samples: Optional[int] = None
    ):
        """
        Initialize lazy dataset.

        Args:
            dataset: HuggingFace dataset
            loss_function: 'mnr', 'cosine', or 'supcon'
            max_samples: Optional limit on samples
        """
        self.dataset = dataset
        self.loss_function = loss_function
        self.max_samples = max_samples if max_samples else len(dataset)

        # For MNR, filter to only positive samples
        if loss_function == 'mnr':
            self.indices = [i for i in range(len(dataset)) if dataset[i]['score'] == 1]
        else:
            self.indices = list(range(len(dataset)))

        # Limit samples
        if max_samples and len(self.indices) > max_samples:
            self.indices = self.indices[:max_samples]

    def __iter__(self):
        """Generate InputExample objects on-the-fly."""
        for idx in self.indices:
            item = self.dataset[idx]

            if self.loss_function in ['mnr', 'cosine']:
                # Both use InputExample format
                if self.loss_function == 'cosine':
                    # Cosine needs label
                    yield InputExample(
                        texts=[item['current_line'], item['context']],
                        label=float(item['score'])
                    )
                else:
                    # MNR no label (all positive)
                    yield InputExample(
                        texts=[item['current_line'], item['context']]
                    )
            else:
                # SupCon uses different format (handled separately)
                yield {
                    'current_line': item['current_line'],
                    'context': item['context'],
                    'score': item['score']
                }

    def __len__(self):
        return len(self.indices)


def create_evaluator(val_dataset: Dataset, name: str = "sql-bug-detection"):
    """
    Create evaluator for validation during training.

    Args:
        val_dataset: Validation dataset
        name: Name for the evaluator

    Returns:
        SentenceTransformer evaluator
    """
    # Create evaluation samples (positive pairs for similarity)
    sentences1 = []
    sentences2 = []
    scores = []

    # Sample up to 1000 examples for evaluation
    eval_size = min(1000, len(val_dataset))

    for i in range(eval_size):
        item = val_dataset[i]
        sentences1.append(item['current_line'])
        sentences2.append(item['context'])
        scores.append(float(item['score']))

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        name=name
    )

    return evaluator


def train_with_mnr(
    model: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainingConfig
):
    """
    Train with MultipleNegativesRankingLoss.

    Args:
        model: SentenceTransformer model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
    """
    print("\n" + "="*80)
    print("TRAINING WITH MULTIPLE NEGATIVES RANKING LOSS (MNR)")
    print("="*80)
    print("Note: Using only POSITIVE pairs (score=1)")

    # Create lazy dataset
    lazy_dataset = LazyInputExampleDataset(
        train_dataset,
        loss_function='mnr',
        max_samples=config.max_train_samples
    )

    print(f"Training samples (positive only): {len(lazy_dataset):,}")

    # Create data loader
    train_dataloader = DataLoader(
        lazy_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Create loss function
    train_loss = st_losses.MultipleNegativesRankingLoss(model)

    # Create evaluator
    evaluator = create_evaluator(val_dataset, name="mnr-validation")

    # Print training summary
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.epochs
    print_training_summary(
        loss_function='mnr',
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        use_lora=config.use_lora,
        use_amp=True,
        use_weighted_sampling=False
    )

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        evaluation_steps=config.eval_steps,
        output_path=config.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\n✓ Training completed! Model saved to: {config.output_dir}")


def train_with_cosine(
    model: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainingConfig
):
    """
    Train with CosineSimilarityLoss.

    Args:
        model: SentenceTransformer model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
    """
    print("\n" + "="*80)
    print("TRAINING WITH COSINE SIMILARITY LOSS")
    print("="*80)
    print("Note: Using ALL data (buggy + correct) with labels")

    # Create lazy dataset
    lazy_dataset = LazyInputExampleDataset(
        train_dataset,
        loss_function='cosine',
        max_samples=config.max_train_samples
    )

    print(f"Training samples: {len(lazy_dataset):,}")

    # Optionally use weighted sampling
    sampler = None
    if config.use_weighted_sampling:
        print("Using weighted sampling for class balance...")
        class_weights, sample_weights = compute_class_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights[:len(lazy_dataset)],
            num_samples=len(lazy_dataset),
            replacement=True
        )

    # Create data loader
    train_dataloader = DataLoader(
        lazy_dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler
    )

    # Create loss function
    train_loss = st_losses.CosineSimilarityLoss(model)

    # Create evaluator
    evaluator = create_evaluator(val_dataset, name="cosine-validation")

    # Print training summary
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.epochs
    print_training_summary(
        loss_function='cosine',
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        use_lora=config.use_lora,
        use_amp=True,
        use_weighted_sampling=config.use_weighted_sampling
    )

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        evaluation_steps=config.eval_steps,
        output_path=config.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\n✓ Training completed! Model saved to: {config.output_dir}")


def train_with_supcon(
    model: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainingConfig
):
    """
    Train with SupervisedContrastiveLoss.

    Note: This is a simplified implementation. For production use,
    consider implementing custom training loop with pytorch_metric_learning.

    Args:
        model: SentenceTransformer model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
    """
    print("\n" + "="*80)
    print("TRAINING WITH SUPERVISED CONTRASTIVE LOSS (SupCon)")
    print("="*80)
    print("Note: Using ALL data (buggy + correct) with contrastive learning")

    # For SupCon, we use a custom training loop
    # This is a simplified version - for full implementation, see train_with_contrastive.py
    print("\nWarning: SupCon implementation is simplified.")
    print("For full contrastive learning, use ensemble loss with supcon component.")

    # Fallback to cosine similarity loss with message
    print("\nFalling back to CosineSimilarityLoss...")
    train_with_cosine(model, train_dataset, val_dataset, config)


def train_with_ensemble(
    model: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainingConfig
):
    """
    Train with weighted ensemble of multiple losses.

    Args:
        model: SentenceTransformer model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
    """
    print("\n" + "="*80)
    print("TRAINING WITH ENSEMBLE LOSS")
    print("="*80)

    # Determine loss types and weights
    loss_types = ['cosine', 'mnr', 'supcon']
    weights = config.ensemble_weights if config.ensemble_weights else [1.0/3, 1.0/3, 1.0/3]

    if len(weights) != len(loss_types):
        print(f"Warning: Expected {len(loss_types)} weights, got {len(weights)}. Using equal weights.")
        weights = [1.0/len(loss_types)] * len(loss_types)

    print(f"Loss types: {loss_types}")
    print(f"Weights: {weights}")

    # Create lazy dataset (uses all data for ensemble)
    lazy_dataset = LazyInputExampleDataset(
        train_dataset,
        loss_function='cosine',  # Ensemble uses all data like cosine
        max_samples=config.max_train_samples
    )

    print(f"Training samples: {len(lazy_dataset):,}")

    # Create data loader
    train_dataloader = DataLoader(
        lazy_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    # Create ensemble loss
    train_loss = WeightedEnsembleLoss(
        model=model,
        loss_types=loss_types,
        weights=weights,
        temperature=config.ensemble_temperature
    )

    # Create evaluator
    evaluator = create_evaluator(val_dataset, name="ensemble-validation")

    # Print training summary
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.epochs
    print_training_summary(
        loss_function='ensemble',
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        use_lora=config.use_lora,
        use_amp=True,
        use_weighted_sampling=False
    )

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        evaluation_steps=config.eval_steps,
        output_path=config.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\n✓ Training completed! Model saved to: {config.output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train SQL bug detection model with multiple loss functions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to HuggingFace dataset directory')
    parser.add_argument('--output-dir', type=str, default='models/sql-bug-detector',
                        help='Output directory for trained model')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='nomic-ai/nomic-embed-code',
                        help='Base model name from HuggingFace')
    parser.add_argument('--loss-function', type=str, default='mnr',
                        choices=['mnr', 'cosine', 'supcon', 'ensemble'],
                        help='Loss function to use (default: mnr)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--eval-steps', type=int, default=2000,
                        help='Evaluation interval in steps')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='Maximum sequence length')

    # Data sampling
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Maximum training samples (for testing)')
    parser.add_argument('--use-weighted-sampling', action='store_true',
                        help='Use weighted sampling for class balance')

    # LoRA arguments
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout')

    # Ensemble arguments
    parser.add_argument('--ensemble-weights', type=float, nargs='+', default=None,
                        help='Weights for ensemble loss [cosine mnr supcon]')
    parser.add_argument('--ensemble-temperature', type=float, default=0.07,
                        help='Temperature for SupCon in ensemble')

    # Misc
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        loss_function=args.loss_function,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        use_weighted_sampling=args.use_weighted_sampling,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        ensemble_weights=args.ensemble_weights,
        ensemble_temperature=args.ensemble_temperature,
        device=args.device,
        seed=args.seed
    )

    # Set device
    if config.device is None:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed
    torch.manual_seed(config.seed)

    # Print configuration
    print_config(vars(config), title="SQL Bug Detection Training Configuration")

    # Load dataset
    print(f"\nLoading dataset from: {config.data_dir}")
    try:
        dataset_dict = load_from_disk(config.data_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nMake sure to run create_sql_dataset.py first to create the HuggingFace dataset.")
        sys.exit(1)

    print(f"  Train: {len(dataset_dict['train']):,} examples")
    print(f"  Validation: {len(dataset_dict['validation']):,} examples")
    print(f"  Test: {len(dataset_dict['test']):,} examples")

    # Load model
    print(f"\nLoading model: {config.model_name}")
    model = SentenceTransformer(config.model_name, device=config.device)
    model.max_seq_length = config.max_seq_length

    # Apply LoRA if requested
    if config.use_lora:
        print("\nApplying LoRA for parameter-efficient fine-tuning...")
        lora_config = setup_lora_config(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )
        # Apply LoRA to the transformer model
        model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
        print("✓ LoRA applied")

    # Train based on loss function
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['validation']

    if config.loss_function == 'mnr':
        train_with_mnr(model, train_dataset, val_dataset, config)
    elif config.loss_function == 'cosine':
        train_with_cosine(model, train_dataset, val_dataset, config)
    elif config.loss_function == 'supcon':
        train_with_supcon(model, train_dataset, val_dataset, config)
    elif config.loss_function == 'ensemble':
        train_with_ensemble(model, train_dataset, val_dataset, config)
    else:
        print(f"Error: Unknown loss function '{config.loss_function}'")
        sys.exit(1)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {config.output_dir}")
    print(f"\nTo use the model:")
    print(f"  from sentence_transformers import SentenceTransformer")
    print(f"  model = SentenceTransformer('{config.output_dir}')")
    print(f"  embeddings = model.encode(['SELECT * FROM users'])")


if __name__ == "__main__":
    main()
