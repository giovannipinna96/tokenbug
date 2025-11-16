"""
Fine-tuning script with selectable contrastive loss functions.

Supports three loss functions:
1. CosineSimilarityLoss - Baseline (uses all data)
2. MultipleNegativesRankingLoss (MNR) - State-of-the-art (uses only positive pairs)
3. SupervisedContrastiveLoss (SupCon) - From SCL-CVD 2024 paper (uses all data)

Usage:
    python train_with_contrastive.py --loss-function mnr --epochs 3 --batch-size 32
    python train_with_contrastive.py --loss-function supcon --temperature 0.07
    python train_with_contrastive.py --loss-function cosine --use-lora
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from pytorch_metric_learning import losses as pml_losses
from datasets import load_from_disk, DatasetDict
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from training_utils import (
    compute_class_weights,
    create_cosine_examples,
    create_mnr_examples,
    create_supcon_data,
    setup_lora_config,
    get_output_dir,
    print_config,
    print_training_summary,
    WeightedEnsembleLoss
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune embedding model with selectable loss function'
    )

    # Loss function selection
    parser.add_argument(
        '--loss-function',
        type=str,
        choices=['cosine', 'mnr', 'supcon', 'ensemble'],
        default='mnr',
        help='Loss function to use: cosine (baseline), mnr (multiple negatives ranking), supcon (supervised contrastive), ensemble (weighted combination)'
    )
    parser.add_argument(
        '--ensemble-losses',
        type=str,
        default='cosine,mnr',
        help='Comma-separated list of losses for ensemble (e.g., "cosine,mnr" or "all" for all three)'
    )
    parser.add_argument(
        '--ensemble-weights',
        type=str,
        default=None,
        help='Comma-separated weights for ensemble losses (e.g., "0.5,0.5"). If not specified, equal weights are used.'
    )

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='nomic-ai/nomic-embed-code',
        help='Pre-trained model to fine-tune'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated from model and loss)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/hf_dataset',
        help='Directory containing the HuggingFace dataset'
    )

    # Training hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: 3e-4 with LoRA, 2e-5 without)'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=1000,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--evaluation-steps',
        type=int,
        default=2000,
        help='Evaluate every N steps'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )

    # LoRA configuration
    parser.add_argument(
        '--use-lora',
        action='store_true',
        default=True,
        help='Use LoRA for parameter-efficient fine-tuning'
    )
    parser.add_argument(
        '--no-lora',
        action='store_false',
        dest='use_lora',
        help='Disable LoRA (full fine-tuning)'
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.1,
        help='LoRA dropout'
    )

    # Data configuration
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=850000,
        help='Maximum training samples (default: 850k = 50%% of dataset)'
    )
    parser.add_argument(
        '--max-val-samples',
        type=int,
        default=780000,
        help='Maximum validation samples (default: 780k = 50%% of dataset)'
    )

    # Loss-specific parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help='Temperature for SupCon loss (default: 0.07)'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.5,
        help='Margin for contrastive losses (default: 0.5)'
    )

    # Other options
    parser.add_argument(
        '--use-gradient-checkpointing',
        action='store_true',
        default=True,
        help='Use gradient checkpointing to save memory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Parse ensemble losses and weights if using ensemble
    if args.loss_function == 'ensemble':
        # Parse loss types
        if args.ensemble_losses.lower() == 'all':
            args.ensemble_loss_types = ['cosine', 'mnr', 'supcon']
        else:
            args.ensemble_loss_types = [l.strip() for l in args.ensemble_losses.split(',')]
            # Validate loss types
            valid_losses = {'cosine', 'mnr', 'supcon'}
            for loss_type in args.ensemble_loss_types:
                if loss_type not in valid_losses:
                    raise ValueError(f"Invalid loss type: {loss_type}. Must be one of {valid_losses}")

        # Parse weights
        if args.ensemble_weights:
            args.ensemble_weight_values = [float(w.strip()) for w in args.ensemble_weights.split(',')]
            if len(args.ensemble_weight_values) != len(args.ensemble_loss_types):
                raise ValueError(f"Number of weights ({len(args.ensemble_weight_values)}) must match number of losses ({len(args.ensemble_loss_types)})")
        else:
            args.ensemble_weight_values = None  # Will use equal weights

    # Set default learning rate based on LoRA usage
    if args.learning_rate is None:
        args.learning_rate = 3e-4 if args.use_lora else 2e-5

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        if args.loss_function == 'ensemble':
            # Create descriptive name for ensemble
            loss_names = '+'.join(args.ensemble_loss_types)
            args.output_dir = str(get_output_dir('./models', args.model_name, f"ensemble-{loss_names}"))
        else:
            args.output_dir = str(get_output_dir('./models', args.model_name, args.loss_function))

    return args


def filter_empty_examples(examples: List[InputExample]) -> List[InputExample]:
    """
    Filter out examples with empty or whitespace-only texts.
    This prevents errors during evaluation when tokenizing empty sequences.

    Args:
        examples: List of InputExample objects

    Returns:
        Filtered list without empty examples
    """
    filtered = []
    removed_count = 0

    for example in examples:
        # Check if any text in the example is empty or whitespace-only
        has_empty = False
        for text in example.texts:
            if not text or not text.strip():
                has_empty = True
                break

        if not has_empty:
            filtered.append(example)
        else:
            removed_count += 1

    if removed_count > 0:
        print(f"⚠️  Filtered out {removed_count} examples with empty texts ({removed_count/len(examples)*100:.2f}%)")

    return filtered


def create_weighted_dataloader(
    examples: List[InputExample],
    sample_weights: List[float],
    batch_size: int
) -> DataLoader:
    """Create DataLoader with weighted sampling."""
    print(f"Creating weighted DataLoader...")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches per epoch: {len(examples) // batch_size:,}")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    dataloader = DataLoader(
        examples,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True
    )

    return dataloader


def train_with_cosine_or_mnr(
    model: SentenceTransformer,
    train_examples: List[InputExample],
    val_examples: List[InputExample],
    loss_function: str,
    args,
    sample_weights: List[float] = None
):
    """
    Train with CosineSimilarityLoss or MultipleNegativesRankingLoss.
    Uses standard sentence-transformers model.fit() API.
    """
    print(f"\n{'=' * 80}")
    print(f"Training with {loss_function.upper()} Loss")
    print(f"{'=' * 80}")

    # Create loss function
    if loss_function == 'cosine':
        train_loss = losses.CosineSimilarityLoss(model)
        print(f"✓ Using CosineSimilarityLoss")
    elif loss_function == 'mnr':
        train_loss = losses.MultipleNegativesRankingLoss(model)
        print(f"✓ Using MultipleNegativesRankingLoss")

    # Create dataloader (with or without weighted sampling)
    if sample_weights and loss_function == 'cosine':
        train_dataloader = create_weighted_dataloader(train_examples, sample_weights, args.batch_size)
        use_weighted = True
    else:
        train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, drop_last=True)
        use_weighted = False

    # Create evaluator (filter out empty examples to prevent tokenization errors)
    val_examples_subset = val_examples[:min(50000, len(val_examples))]
    val_examples_filtered = filter_empty_examples(val_examples_subset)
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples_filtered,
        name='validation',
        batch_size=args.batch_size
    )
    print(f"✓ Validation evaluator created with {len(val_examples_filtered):,} examples")

    # Print training summary
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs
    print_training_summary(
        loss_function=loss_function,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        use_amp=True,
        use_weighted_sampling=use_weighted
    )

    # Training
    print(f"\nTraining started...")
    print("-" * 80)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
        output_path=args.output_dir,
        use_amp=True,
        optimizer_params={'lr': args.learning_rate},
        show_progress_bar=True,
        save_best_model=True,
        checkpoint_save_steps=args.evaluation_steps,
        checkpoint_save_total_limit=3,
    )

    print("-" * 80)
    print(f"✓ Training complete!")


def train_with_supcon(
    model: SentenceTransformer,
    lines: List[str],
    contexts: List[str],
    labels: List[int],
    val_lines: List[str],
    val_contexts: List[str],
    val_labels: List[int],
    args,
    sample_weights: List[float]
):
    """
    Train with Supervised Contrastive Loss using pytorch-metric-learning.
    Uses custom training loop.
    """
    print(f"\n{'=' * 80}")
    print(f"Training with SUPCON (Supervised Contrastive) Loss")
    print(f"{'=' * 80}")

    # Create SupCon loss
    supcon_loss = pml_losses.SupConLoss(temperature=args.temperature)
    print(f"✓ Using SupConLoss (temperature={args.temperature})")

    # Prepare data
    # Combine lines and contexts for embedding
    train_texts = []
    for line, context in zip(lines, contexts):
        train_texts.append(line)  # We'll embed line and context separately then combine
        train_texts.append(context)

    # Create weighted sampler for batches
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create dataset indices
    indices = list(range(len(lines)))
    batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Setup scheduler
    steps_per_epoch = len(sample_weights) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps
    )

    print_training_summary(
        loss_function='supcon',
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        use_amp=True,
        use_weighted_sampling=True
    )

    # Training loop
    print(f"\nTraining started...")
    print("-" * 80)

    model.train()
    device = model.device
    global_step = 0
    best_val_score = -1

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0

        # Create progress bar
        pbar = tqdm(batch_sampler, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_indices in pbar:
            # Get batch data
            batch_lines = [lines[i] for i in batch_indices]
            batch_contexts = [contexts[i] for i in batch_indices]
            batch_labels_list = [labels[i] for i in batch_indices]

            # Filter out empty strings to prevent tokenization errors
            valid_pairs = []
            for i, (line, context, label) in enumerate(zip(batch_lines, batch_contexts, batch_labels_list)):
                if line and line.strip() and context and context.strip():
                    valid_pairs.append((line, context, label))

            # Skip batch if all examples are invalid
            if not valid_pairs:
                continue

            # Unpack valid pairs
            batch_lines = [p[0] for p in valid_pairs]
            batch_contexts = [p[1] for p in valid_pairs]
            batch_labels = torch.tensor([p[2] for p in valid_pairs], device=device)

            # Tokenize and encode lines and contexts with gradient tracking
            with torch.amp.autocast('cuda'):
                # Tokenize inputs
                line_features = model.tokenize(batch_lines)
                context_features = model.tokenize(batch_contexts)

                # Move to device
                line_features = {k: v.to(device) for k, v in line_features.items()}
                context_features = {k: v.to(device) for k, v in context_features.items()}

                # Forward pass through model (maintains gradients)
                line_outputs = model(line_features)
                context_outputs = model(context_features)

                # Extract embeddings (sentence_embedding key for SentenceTransformer)
                line_embeddings = line_outputs['sentence_embedding']
                context_embeddings = context_outputs['sentence_embedding']

                # Combine embeddings (average pooling)
                combined_embeddings = (line_embeddings + context_embeddings) / 2
                combined_embeddings = F.normalize(combined_embeddings, p=2, dim=1)

                # Compute loss
                loss = supcon_loss(combined_embeddings, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step < args.warmup_steps:
                warmup_scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Evaluation
            if global_step % args.evaluation_steps == 0:
                val_score = evaluate_supcon(model, val_lines[:10000], val_contexts[:10000], val_labels[:10000], device)
                print(f"\nStep {global_step}: Validation score: {val_score:.4f}")

                if val_score > best_val_score:
                    best_val_score = val_score
                    print(f"✓ New best model! Saving to {args.output_dir}")
                    model.save(args.output_dir)

                model.train()

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")

    print("-" * 80)
    print(f"✓ Training complete!")

    # Final save
    print(f"\nSaving final model to {args.output_dir}")
    model.save(args.output_dir)


def evaluate_supcon(model, lines, contexts, labels, device):
    """Simple evaluation for SupCon: compute embedding similarity correlation with labels."""
    model.eval()

    # Filter out empty strings to prevent tokenization errors
    valid_indices = []
    for i, (line, context) in enumerate(zip(lines, contexts)):
        if line and line.strip() and context and context.strip():
            valid_indices.append(i)

    if not valid_indices:
        print("⚠️  Warning: All validation examples have empty strings, skipping evaluation")
        model.train()
        return 0.0

    # Filter data
    filtered_lines = [lines[i] for i in valid_indices]
    filtered_contexts = [contexts[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]

    with torch.no_grad():
        line_embeddings = model.encode(filtered_lines, convert_to_tensor=True, show_progress_bar=False, device=device)
        context_embeddings = model.encode(filtered_contexts, convert_to_tensor=True, show_progress_bar=False, device=device)

        # Cosine similarity
        similarities = F.cosine_similarity(line_embeddings, context_embeddings, dim=1)

        # Spearman correlation with labels
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(similarities.cpu().numpy(), filtered_labels)

    model.train()
    return correlation


def train_with_ensemble(
    model: SentenceTransformer,
    train_examples: List[InputExample],
    val_examples: List[InputExample],
    args,
    sample_weights: List[float]
):
    """
    Train with weighted ensemble of multiple loss functions.
    Uses custom training loop with WeightedEnsembleLoss.
    """
    print(f"\n{'=' * 80}")
    print(f"Training with ENSEMBLE Loss")
    print(f"{'=' * 80}")

    # Create weighted ensemble loss
    ensemble_loss = WeightedEnsembleLoss(
        model=model,
        loss_types=args.ensemble_loss_types,
        weights=args.ensemble_weight_values,
        temperature=args.temperature
    )

    # Create dataloader with weighted sampling
    # Use all data (both buggy and correct)
    train_dataloader = create_weighted_dataloader(train_examples, sample_weights, args.batch_size)
    use_weighted = True

    # Create evaluator (filter out empty examples to prevent tokenization errors)
    val_examples_subset = val_examples[:min(50000, len(val_examples))]
    val_examples_filtered = filter_empty_examples(val_examples_subset)
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples_filtered,
        name='validation',
        batch_size=args.batch_size
    )
    print(f"✓ Validation evaluator created with {len(val_examples_filtered):,} examples")

    # Print training summary
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs

    print(f"\n{'=' * 80}")
    print(f"ENSEMBLE TRAINING SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nLoss Functions: {', '.join([l.upper() for l in args.ensemble_loss_types])}")
    if args.ensemble_weight_values:
        print(f"Weights: {', '.join([f'{w:.3f}' for w in args.ensemble_weight_values])}")
    else:
        print(f"Weights: Equal (auto-balanced)")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"\nOptimizations:")
    print(f"  LoRA: {'Yes' if args.use_lora else 'No'}")
    print(f"  AMP (Mixed Precision): Yes")
    print(f"  Weighted Sampling: {use_weighted}")
    print(f"\n{'=' * 80}")

    # Training
    print(f"\nTraining started...")
    print("-" * 80)

    model.fit(
        train_objectives=[(train_dataloader, ensemble_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
        output_path=args.output_dir,
        use_amp=True,
        optimizer_params={'lr': args.learning_rate},
        show_progress_bar=True,
        save_best_model=True,
        checkpoint_save_steps=args.evaluation_steps,
        checkpoint_save_total_limit=3,
    )

    print("-" * 80)
    print(f"✓ Training complete!")

    # Print final loss breakdown if available
    if hasattr(ensemble_loss, 'last_individual_losses'):
        print(f"\nFinal individual losses:")
        for loss_type, loss_value in ensemble_loss.last_individual_losses.items():
            print(f"  {loss_type.upper()}: {loss_value:.4f}")


def main():
    """Main training function."""
    args = parse_args()

    # Print configuration
    config_dict = vars(args)
    print_config(config_dict, "Training Configuration")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'=' * 80}")
    print(f"Device Setup")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model
    print(f"\n{'=' * 80}")
    print(f"Loading Model")
    print(f"{'=' * 80}")
    print(f"Model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = args.max_seq_length
    print(f"✓ Model loaded")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Apply LoRA if requested
    if args.use_lora:
        print(f"\n{'=' * 80}")
        print(f"Applying LoRA")
        print(f"{'=' * 80}")
        lora_config = setup_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
        model.add_adapter(lora_config)

        if args.use_gradient_checkpointing:
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ LoRA adapter applied")
        print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total parameters: {total_params:,}")

    # Load dataset
    print(f"\n{'=' * 80}")
    print(f"Loading Dataset")
    print(f"{'=' * 80}")
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run create_hf_dataset_optimized.py first.")

    dataset = load_from_disk(str(data_path))
    print(f"✓ Dataset loaded")

    # Handle splits
    if 'validation' not in dataset or 'test' not in dataset:
        print(f"\n⚠ Missing splits. Creating from train split...")
        train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        temp_test = train_test_split['test']
        val_test_split = temp_test.train_test_split(test_size=0.5, seed=42)
        validation_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
        dataset = DatasetDict({'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset})

    print(f"  Train: {len(dataset['train']):,}")
    print(f"  Validation: {len(dataset['validation']):,}")
    print(f"  Test: {len(dataset['test']):,}")

    # Compute class weights
    class_weights, train_sample_weights = compute_class_weights(dataset['train'])

    # Prepare data based on loss function
    print(f"\n{'=' * 80}")
    print(f"Preparing Data for {args.loss_function.upper()} Loss")
    print(f"{'=' * 80}")

    if args.loss_function == 'cosine':
        train_examples = create_cosine_examples(dataset['train'], args.max_train_samples)
        val_examples = create_cosine_examples(dataset['validation'], args.max_val_samples)
        if args.max_train_samples:
            train_sample_weights = train_sample_weights[:args.max_train_samples]
        train_with_cosine_or_mnr(model, train_examples, val_examples, 'cosine', args, train_sample_weights)

    elif args.loss_function == 'mnr':
        train_examples = create_mnr_examples(dataset['train'], args.max_train_samples)
        val_examples = create_mnr_examples(dataset['validation'], args.max_val_samples)
        # MNR doesn't use weighted sampling (only positive pairs)
        train_with_cosine_or_mnr(model, train_examples, val_examples, 'mnr', args, sample_weights=None)

    elif args.loss_function == 'supcon':
        lines, contexts, labels = create_supcon_data(dataset['train'], args.max_train_samples)
        val_lines, val_contexts, val_labels = create_supcon_data(dataset['validation'], args.max_val_samples)
        if args.max_train_samples:
            train_sample_weights = train_sample_weights[:args.max_train_samples]
        train_with_supcon(model, lines, contexts, labels, val_lines, val_contexts, val_labels, args, train_sample_weights)

    elif args.loss_function == 'ensemble':
        # For ensemble, use all data (like cosine/supcon)
        train_examples = create_cosine_examples(dataset['train'], args.max_train_samples)
        val_examples = create_cosine_examples(dataset['validation'], args.max_val_samples)
        if args.max_train_samples:
            train_sample_weights = train_sample_weights[:args.max_train_samples]
        train_with_ensemble(model, train_examples, val_examples, args, train_sample_weights)

    # Final evaluation
    print(f"\n{'=' * 80}")
    print(f"Final Evaluation")
    print(f"{'=' * 80}")
    print(f"Model saved to: {args.output_dir}")
    print(f"\nTo load the fine-tuned model:")
    print(f"  from sentence_transformers import SentenceTransformer")
    print(f"  model = SentenceTransformer('{args.output_dir}')")
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
