"""
Fine-tuning script for nomic-ai/nomic-embed-code on bug detection task with LoRA.
MULTI-GPU VERSION with Accelerate support.

This script uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
of large embedding models to learn similarity between code lines and their context:
- Cosine similarity ≈ 1.0 for correct lines (score=1)
- Cosine similarity ≈ 0.0 for buggy lines (score=0)

Multi-GPU optimizations:
- Distributed Data Parallel (DDP) with 4x A100 60GB
- Per-GPU batch size: 64 (Effective total: 256 across 4 GPUs)
- Expected speedup: ~3.5x compared to single GPU
- Training time: ~85 hours (vs ~300 hours single GPU)

LoRA advantages:
- Only ~0.1% of parameters are trainable
- Dramatically reduced memory requirements
- Compatible with multi-GPU training

Launch with:
    accelerate launch --config_file accelerate_config.yaml train_embedding_model_multigpu.py
"""

import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from peft import LoraConfig, TaskType, get_peft_model_state_dict
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
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader with weighted sampling for class balance.

    Args:
        examples: List of InputExample objects
        sample_weights: Weight for each sample
        batch_size: Batch size for training
        shuffle: Whether to shuffle (ignored if using weighted sampling)

    Returns:
        DataLoader with weighted sampling
    """
    print(f"Creating weighted DataLoader...")
    print(f"  Batch size per GPU: {batch_size}")
    print(f"  Total batches per epoch: {len(examples) // batch_size:,}")

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
        drop_last=True  # Drop incomplete batches for consistent batch size
    )

    return dataloader


def main():
    """Main training function."""

    # ==================== Multi-GPU Environment Info ====================
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_main_process = local_rank in [-1, 0]

    if is_main_process:
        print("=" * 80)
        print("Fine-tuning nomic-embed-code for Bug Detection (MULTI-GPU)")
        print("=" * 80)
        print(f"\nMulti-GPU Configuration:")
        print(f"  World size (total GPUs): {world_size}")
        print(f"  Local rank: {local_rank}")
        print(f"  Is main process: {is_main_process}")

    # ==================== Configuration ====================
    MODEL_NAME = 'nomic-ai/nomic-embed-code'
    OUTPUT_DIR = Path('./models/finetuned-nomic-embed-multigpu')
    DATA_DIR = Path('./data/hf_dataset')

    # LoRA Configuration for parameter-efficient fine-tuning
    USE_LORA = True
    LORA_CONFIG = {
        'r': 16,  # LoRA rank - higher = more capacity but more memory
        'lora_alpha': 32,  # LoRA scaling factor (typically 2*r)
        'lora_dropout': 0.1,  # Dropout for LoRA layers
        'target_modules': ['q_proj', 'k_proj', 'v_proj'],  # Apply LoRA to Q, K, V attention projections
        'task_type': TaskType.FEATURE_EXTRACTION,
        'inference_mode': False,
    }

    # Training hyperparameters (optimized for 4x A100 60GB with LoRA)
    BATCH_SIZE = 64  # Per-GPU batch size (Effective total: 64 * 4 = 256)
    NUM_EPOCHS = 3
    WARMUP_STEPS = 1000
    LEARNING_RATE = 3e-4  # Higher LR for LoRA (recommended: 1e-4 to 5e-4)
    EVALUATION_STEPS = 2000
    MAX_SEQ_LENGTH = 512  # Full sequence length with LoRA
    USE_GRADIENT_CHECKPOINTING = True  # Further memory optimization

    # Use 50% of dataset - best compromise between data size and memory
    MAX_TRAIN_SAMPLES = 850000  # 50% of 1,704,039
    MAX_VAL_SAMPLES = 780000    # 50% of 1,565,841

    if is_main_process:
        print(f"\nConfiguration:")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  Data directory: {DATA_DIR}")
        print(f"  Use LoRA: {USE_LORA}")
        if USE_LORA:
            print(f"  LoRA rank (r): {LORA_CONFIG['r']}")
            print(f"  LoRA alpha: {LORA_CONFIG['lora_alpha']}")
            print(f"  LoRA dropout: {LORA_CONFIG['lora_dropout']}")
            print(f"  LoRA target modules: {LORA_CONFIG['target_modules']}")
        print(f"  Batch size per GPU: {BATCH_SIZE}")
        print(f"  Effective batch size (total): {BATCH_SIZE * world_size}")
        print(f"  Epochs: {NUM_EPOCHS}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Warmup steps: {WARMUP_STEPS}")
        print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
        print(f"  Gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
        print(f"  Dataset size: {MAX_TRAIN_SAMPLES:,} train / {MAX_VAL_SAMPLES:,} val (50%)")

    # ==================== Device Setup ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if is_main_process:
        print(f"\nDevice: {device}")
        if device == 'cuda':
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
            print(f"  Memory per GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  Total GPUs: {world_size}")
            print(f"  Total VRAM: {world_size * torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ==================== Load Model ====================
    if is_main_process:
        print(f"\nLoading model: {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    if is_main_process:
        print(f"✓ Model loaded")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ==================== Apply LoRA ====================
    if USE_LORA:
        if is_main_process:
            print(f"\nApplying LoRA adapter...")

        # Create LoRA configuration
        lora_config = LoraConfig(**LORA_CONFIG)

        # Add LoRA adapter to the model
        model.add_adapter(lora_config)

        # Enable gradient checkpointing if requested
        if USE_GRADIENT_CHECKPOINTING:
            if is_main_process:
                print(f"  Enabling gradient checkpointing...")
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()

        # Print trainable parameters info
        if is_main_process:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ LoRA adapter applied")
            print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Memory savings: ~{100 * (1 - trainable_params / total_params):.1f}% reduction in optimizer states")

    # ==================== Load Dataset ====================
    if is_main_process:
        print(f"\nLoading dataset from {DATA_DIR}...")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}. "
            "Please run create_hf_dataset.py first."
        )

    dataset = load_from_disk(str(DATA_DIR))

    if is_main_process:
        print(f"✓ Dataset loaded")
        print(f"  Train samples: {len(dataset['train']):,}")
        print(f"  Validation samples: {len(dataset['validation']):,}")
        print(f"  Test samples: {len(dataset['test']):,}")

    # ==================== Compute Class Weights ====================
    if is_main_process:
        class_weights, train_sample_weights = compute_class_weights(dataset['train'])
    else:
        # Non-main processes still need these for data loading
        scores = [item['score'] for item in dataset['train']]
        class_counts = Counter(scores)
        total_samples = len(scores)
        class_weights = np.array([
            total_samples / (len(class_counts) * class_counts[0]),
            total_samples / (len(class_counts) * class_counts[1])
        ])
        train_sample_weights = [class_weights[score] for score in scores]

    # ==================== Prepare Training Data ====================
    if is_main_process:
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
        shuffle=True
    )

    # ==================== Prepare Validation Data ====================
    if is_main_process:
        print(f"\nPreparing validation data...")

    val_examples = create_examples(dataset['validation'], max_samples=MAX_VAL_SAMPLES)

    # Create evaluator for validation
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='validation',
        batch_size=BATCH_SIZE
    )

    if is_main_process:
        print(f"✓ Validation evaluator created")

    # ==================== Define Loss Function ====================
    if is_main_process:
        print(f"\nConfiguring loss function...")

    train_loss = losses.CosineSimilarityLoss(model)

    if is_main_process:
        print(f"✓ Using CosineSimilarityLoss")
        print(f"  This loss optimizes cosine similarity to match target scores (0.0 or 1.0)")

    # ==================== Training ====================
    if is_main_process:
        print(f"\n{'=' * 80}")
        print("Starting Multi-GPU Training")
        print("=" * 80)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * NUM_EPOCHS

    if is_main_process:
        print(f"\nTraining details:")
        print(f"  Steps per epoch (per GPU): {steps_per_epoch:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Effective batch size: {BATCH_SIZE * world_size}")
        print(f"  Warmup steps: {WARMUP_STEPS}")
        print(f"  Evaluation every: {EVALUATION_STEPS} steps")
        print(f"  Optimizer: AdamW (default)")
        print(f"  Using Automatic Mixed Precision (AMP): Yes")
        print(f"  Using Weighted Sampling: Yes")
        print(f"  Using LoRA: {USE_LORA}")
        print(f"  Using Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
        print(f"  Multi-GPU: DDP with {world_size} GPUs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if is_main_process:
        print(f"\nTraining started...")
        print("-" * 80)

    # sentence-transformers.fit() handles DDP automatically when launched with accelerate/torchrun
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        evaluator=evaluator,
        evaluation_steps=EVALUATION_STEPS,
        output_path=str(OUTPUT_DIR),
        use_amp=True,  # Enable Automatic Mixed Precision
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=is_main_process,  # Only main process shows progress
        save_best_model=True,
        checkpoint_save_steps=EVALUATION_STEPS,
        checkpoint_save_total_limit=3,
    )

    if is_main_process:
        print("-" * 80)
        print(f"✓ Training complete!")

    # ==================== Final Evaluation ====================
    if is_main_process:
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
        print(f"\nMulti-GPU Training:")
        print(f"  GPUs used: {world_size}")
        print(f"  Effective batch size: {BATCH_SIZE * world_size}")
        print(f"\nTo load the fine-tuned model:")
        print(f"  from sentence_transformers import SentenceTransformer")
        print(f"  model = SentenceTransformer('{OUTPUT_DIR}')")
        print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
