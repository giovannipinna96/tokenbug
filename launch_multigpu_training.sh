#!/bin/bash
# Launch script for multi-GPU training with Accelerate
# For 4x A100 60GB GPUs training nomic-embed-code with LoRA

set -e  # Exit on error

echo "=============================================================================="
echo "Multi-GPU Training Launcher for nomic-embed-code with LoRA"
echo "=============================================================================="
echo ""

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "❌ Error: accelerate is not installed"
    echo "Install with: pip install accelerate"
    exit 1
fi

# Check if config file exists
if [ ! -f "accelerate_config.yaml" ]; then
    echo "❌ Error: accelerate_config.yaml not found"
    echo "Please create the config file first"
    exit 1
fi

# Check if training script exists
if [ ! -f "train_embedding_model_multigpu.py" ]; then
    echo "❌ Error: train_embedding_model_multigpu.py not found"
    exit 1
fi

# Check number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "✓ Detected $NUM_GPUS GPUs"
echo ""

if [ "$NUM_GPUS" -lt 4 ]; then
    echo "⚠️  Warning: Less than 4 GPUs detected"
    echo "   Config is optimized for 4 GPUs"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display configuration
echo "Training Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Config file: accelerate_config.yaml"
echo "  - Training script: train_embedding_model_multigpu.py"
echo "  - Per-GPU batch size: 64"
echo "  - Effective batch size: $(($NUM_GPUS * 64))"
echo "  - Mixed precision: FP16"
echo ""

# Ask for confirmation
read -p "Start training? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo "=============================================================================="
echo "Starting Multi-GPU Training..."
echo "=============================================================================="
echo ""

# Launch with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    train_embedding_model_multigpu.py

echo ""
echo "=============================================================================="
echo "Training finished!"
echo "=============================================================================="
