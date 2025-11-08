"""
Fine-tuning script for st-codesearch-distilroberta-base on bug detection task.

This script fine-tunes a lightweight DistilRoBERTa embedding model (82M parameters)
to learn similarity between code lines and their context:
- Cosine similarity ≈ 1.0 for correct lines (score=1)
- Cosine similarity ≈ 0.0 for buggy lines (score=0)

Model advantages:
- Compact size (82M params vs 7B for nomic-embed-code)
- Fast training and inference
- Already pre-trained on code search tasks
- Can use full dataset with high batch size

Optimized for A100 80GB GPU with:
- Large batch size (128) - full GPU utilization
- Full dataset training (1.7M samples)
- Automatic Mixed Precision (AMP)
- Weighted sampling for class imbalance
- Efficient data loading
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from datasets import load_from_disk, DatasetDict
from pathlib import Path
from typing import List, Tuple
import numpy as np
from collections import Counter


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


def create_examples(dataset, max_samples: int = None) -> List[InputExample]:
    """
    Convert HuggingFace dataset to sentence-transformers InputExample format.

    Args:
        dataset: HuggingFace dataset with current_line, context, and score
        max_samples: Optional limit on number of samples (for testing)

    Returns:
        List of InputExample objects
    """
    print(f"Converting dataset to InputExample format...")
    examples = []

    dataset_slice = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    for idx, item in enumerate(dataset_slice):
        if idx % 50000 == 0:
            print(f"  Processed {idx:,}/{len(dataset_slice):,} examples")

        # Create InputExample with (text1, text2, similarity_score)
        example = InputExample(
            texts=[item['current_line'], item['context']],
            label=float(item['score'])  # 0.0 or 1.0
        )
        examples.append(example)

    print(f"✓ Created {len(examples):,} training examples")
    return examples


def create_weighted_dataloader(
    examples: List[InputExample],
    sample_weights: List[float],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader with weighted sampling for class balance.

    Args:
        examples: List of InputExample objects
        sample_weights: Weight for each sample
        batch_size: Batch size for training
        shuffle: Whether to shuffle (ignored if using weighted sampling)
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader with weighted sampling
    """
    print(f"Creating weighted DataLoader...")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches per epoch: {len(examples) // batch_size:,}")
    print(f"  Num workers: {num_workers}")

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow sampling with replacement
    )

    # Create DataLoader with sampler (shuffle must be False when using sampler)
    dataloader = DataLoader(
        examples,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,  # Drop incomplete batches for consistent batch size
        num_workers=num_workers,
        pin_memory=True  # Speed up GPU transfer
    )

    return dataloader


def main():
    """Main training function."""

    # ==================== Configuration ====================
    MODEL_NAME = 'flax-sentence-embeddings/st-codesearch-distilroberta-base'
    OUTPUT_DIR = Path('./models/finetuned-distilroberta')
    DATA_DIR = Path('./data/hf_dataset')

    # Training hyperparameters (optimized for A100 80GB + small model)
    BATCH_SIZE = 128  # Large batch size - model is only 82M params
    NUM_EPOCHS = 3
    WARMUP_STEPS = 1000  # More warmup for large dataset
    LEARNING_RATE = 2e-5  # Standard fine-tuning LR
    EVALUATION_STEPS = 2000  # Evaluate less frequently (large dataset)
    MAX_SEQ_LENGTH = 512  # DistilRoBERTa supports up to 512
    NUM_WORKERS = 0  # Disabled to avoid deadlock with WeightedRandomSampler

    # Use 50% of dataset - best compromise between data size and memory
    # This uses ~10GB RAM which is safe, vs 20GB for full dataset
    MAX_TRAIN_SAMPLES = 850000  # 50% of 1,704,039
    MAX_VAL_SAMPLES = 780000    # 50% of 1,565,841

    print("=" * 80)
    print("Fine-tuning st-codesearch-distilroberta-base for Bug Detection")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Dataset size: {MAX_TRAIN_SAMPLES:,} train / {MAX_VAL_SAMPLES:,} val (50% - optimal balance)")

    # ==================== Device Setup ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ==================== Load Model ====================
    print(f"\nLoading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"✓ Model loaded")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e9:.2f} GB (FP32)")

    # ==================== Load Dataset ====================
    print(f"\nLoading dataset from {DATA_DIR}...")
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}. "
            "Please run create_hf_dataset.py first."
        )

    dataset = load_from_disk(str(DATA_DIR))
    print(f"✓ Dataset loaded")

    # Check if validation and test splits exist, if not create them from train
    if 'validation' not in dataset or 'test' not in dataset:
        print(f"\n⚠ Missing validation or test splits. Creating them from train split...")
        print(f"  Original train samples: {len(dataset['train']):,}")

        # Split ratios: 80% train, 10% validation, 10% test
        train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        temp_test = train_test_split['test']

        # Split the test portion into validation and test (50/50)
        val_test_split = temp_test.train_test_split(test_size=0.5, seed=42)
        validation_dataset = val_test_split['train']
        test_dataset = val_test_split['test']

        # Create new DatasetDict with all splits
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset
        })

        print(f"✓ Splits created:")
        print(f"    Train: {len(dataset['train']):,} samples (80%)")
        print(f"    Validation: {len(dataset['validation']):,} samples (10%)")
        print(f"    Test: {len(dataset['test']):,} samples (10%)")
    else:
        print(f"  Train samples: {len(dataset['train']):,}")
        print(f"  Validation samples: {len(dataset['validation']):,}")
        print(f"  Test samples: {len(dataset['test']):,}")

    # ==================== Compute Class Weights ====================
    class_weights, train_sample_weights = compute_class_weights(dataset['train'])

    # ==================== Prepare Training Data ====================
    print(f"\nPreparing training data...")
    train_examples = create_examples(dataset['train'], max_samples=MAX_TRAIN_SAMPLES)

    # If we limited samples, also limit weights
    if MAX_TRAIN_SAMPLES is not None:
        train_sample_weights = train_sample_weights[:MAX_TRAIN_SAMPLES]

    # Create weighted DataLoader
    train_dataloader = create_weighted_dataloader(
        train_examples,
        train_sample_weights,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # ==================== Prepare Validation Data ====================
    print(f"\nPreparing validation data...")
    val_examples = create_examples(dataset['validation'], max_samples=MAX_VAL_SAMPLES)

    # Create evaluator for validation
    # EmbeddingSimilarityEvaluator computes cosine similarity and compares to labels
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='validation',
        batch_size=BATCH_SIZE
    )
    print(f"✓ Validation evaluator created")

    # ==================== Define Loss Function ====================
    print(f"\nConfiguring loss function...")
    train_loss = losses.CosineSimilarityLoss(model)
    print(f"✓ Using CosineSimilarityLoss")
    print(f"  This loss optimizes cosine similarity to match target scores (0.0 or 1.0)")

    # ==================== Training ====================
    print(f"\n{'=' * 80}")
    print("Starting Training")
    print("=" * 80)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"\nTraining details:")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Evaluation every: {EVALUATION_STEPS} steps")
    print(f"  Optimizer: AdamW")
    print(f"  Using Automatic Mixed Precision (AMP): Yes")
    print(f"  Using Weighted Sampling: Yes")
    print(f"  Using Full Fine-Tuning: Yes (no LoRA)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining started...")
    print("-" * 80)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        evaluator=evaluator,
        evaluation_steps=EVALUATION_STEPS,
        output_path=str(OUTPUT_DIR),
        use_amp=True,  # Enable Automatic Mixed Precision
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True,
        save_best_model=True,
        checkpoint_save_steps=EVALUATION_STEPS,
        checkpoint_save_total_limit=3,  # Keep only 3 best checkpoints
    )

    print("-" * 80)
    print(f"✓ Training complete!")

    # ==================== Final Evaluation ====================
    print(f"\n{'=' * 80}")
    print("Final Evaluation")
    print("=" * 80)

    # Evaluate on validation set
    print(f"\nEvaluating on validation set...")
    val_score = evaluator(model, output_path=str(OUTPUT_DIR))
    print(f"✓ Validation correlation: {val_score:.4f}")

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_examples = create_examples(dataset['test'])
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        test_examples,
        name='test',
        batch_size=BATCH_SIZE
    )
    test_score = test_evaluator(model, output_path=str(OUTPUT_DIR))
    print(f"✓ Test correlation: {test_score:.4f}")

    # ==================== Summary ====================
    print(f"\n{'=' * 80}")
    print("Training Summary")
    print("=" * 80)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Validation correlation: {val_score:.4f}")
    print(f"Test correlation: {test_score:.4f}")
    print(f"\nTo load the fine-tuned model:")
    print(f"  from sentence_transformers import SentenceTransformer")
    print(f"  model = SentenceTransformer('{OUTPUT_DIR}')")
    print(f"\nTo use for inference:")
    print(f"  line_embedding = model.encode('your code line')")
    print(f"  context_embedding = model.encode('your code context')")
    print(f"  similarity = cosine_similarity([line_embedding], [context_embedding])[0][0]")
    print(f"  is_buggy = similarity < 0.5  # threshold at 0.5")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
