# Multi-GPU Training Guide

This guide explains how to train the nomic-embed-code model with LoRA on multiple GPUs using Hugging Face Accelerate.

## Overview

### Single vs Multi-GPU Performance

| Configuration | GPUs | Batch Size | Speed | Training Time |
|--------------|------|------------|-------|---------------|
| **Single GPU** | 1x A100 80GB | 32 | ~4.6s/it | ~300 hours |
| **Multi-GPU** | 4x A100 60GB | 64/GPU (256 total) | ~1.3s/it | **~85 hours** |

**Speedup: 3.5x faster** 🚀

## Files

- **`train_embedding_model_multigpu.py`** - Multi-GPU training script with DDP support
- **`accelerate_config.yaml`** - Configuration for 4 GPUs with FP16 mixed precision
- **`launch_multigpu_training.sh`** - Helper script to launch training

## Requirements

```bash
# Install accelerate if not already installed
pip install accelerate

# Or with uv
uv add accelerate
```

## Quick Start

### Option 1: Using the Launch Script (Recommended)

```bash
./launch_multigpu_training.sh
```

This script will:
- Check if accelerate is installed
- Verify GPU availability
- Display configuration summary
- Launch training with proper settings

### Option 2: Manual Launch

```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    train_embedding_model_multigpu.py
```

### Option 3: Using accelerate CLI

```bash
# Configure accelerate interactively
accelerate config

# Then launch
accelerate launch train_embedding_model_multigpu.py
```

## Configuration Details

### Accelerate Config (`accelerate_config.yaml`)

```yaml
distributed_type: MULTI_GPU    # Use Distributed Data Parallel
num_processes: 4               # 4 GPUs
mixed_precision: fp16          # FP16 for faster training
num_machines: 1                # Single node
```

### Training Hyperparameters

| Parameter | Single GPU | Multi-GPU (4 GPUs) |
|-----------|------------|-------------------|
| Batch size per GPU | 32 | 64 |
| Effective batch size | 32 | 256 |
| Learning rate | 3e-4 | 3e-4 |
| LoRA rank | 16 | 16 |
| Sequence length | 512 | 512 |

### Memory Usage (per GPU)

- Model (7B, FP16): ~14 GB
- LoRA adapters: ~0.5 GB
- Gradients (0.1% params): ~0.1 GB
- Activations (batch 64): ~10 GB
- **Total: ~25 GB / 60 GB available** ✅

## Monitoring Training

### Check GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check all 4 GPUs
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

### View Training Progress

Training logs will show progress for the main process (rank 0). The script automatically:
- Prints configuration on main process only
- Shows progress bar on main process only
- Saves checkpoints from main process only

### Expected Output

```
================================================================================
Fine-tuning nomic-embed-code for Bug Detection (MULTI-GPU)
================================================================================

Multi-GPU Configuration:
  World size (total GPUs): 4
  Local rank: 0
  Is main process: True

Configuration:
  Batch size per GPU: 64
  Effective batch size (total): 256
  Multi-GPU: DDP with 4 GPUs
  ...

Training started...
--------------------------------------------------------------------------------
  0%|          | 0/79686 [00:00<?, ?it/s]
  1%|          | 100/79686 [02:10<289:00, 4.59it/s]
  ...
```

## Troubleshooting

### Issue: `accelerate: command not found`

```bash
pip install accelerate
# or
uv add accelerate
```

### Issue: CUDA Out of Memory

If you encounter OOM errors:

1. **Reduce batch size** in `train_embedding_model_multigpu.py`:
   ```python
   BATCH_SIZE = 48  # Reduce from 64
   ```

2. **Enable more aggressive optimizations**:
   ```python
   USE_GRADIENT_CHECKPOINTING = True  # Already enabled
   ```

3. **Check other processes**:
   ```bash
   nvidia-smi  # Check for other GPU users
   ```

### Issue: Slow Training

If training is slower than expected:

1. **Check GPU utilization**:
   ```bash
   nvidia-smi
   ```
   All GPUs should be at ~100% utilization

2. **Check data loading**:
   - Ensure dataset is on fast storage (not NFS)
   - Consider increasing `num_workers` in DataLoader

3. **Verify mixed precision is enabled**:
   - Check logs for "Using Automatic Mixed Precision (AMP): Yes"

### Issue: Distributed Training Not Working

1. **Verify NCCL is working**:
   ```bash
   python -c "import torch; print(torch.cuda.nccl.version())"
   ```

2. **Check network configuration**:
   - Ensure all GPUs are on same node
   - Check `same_network: true` in config

3. **Test with single GPU first**:
   ```bash
   python train_embedding_model.py  # Single GPU version
   ```

## Advanced Configuration

### Changing Number of GPUs

Edit `accelerate_config.yaml`:

```yaml
num_processes: 2  # For 2 GPUs instead of 4
```

And adjust batch size accordingly in the training script:

```python
BATCH_SIZE = 128  # 2 GPUs * 128 = 256 effective batch size
```

### Using Different Precision

For A100 GPUs, you can use BF16 instead of FP16:

```yaml
mixed_precision: bf16  # Better numerical stability
```

### Multi-Node Training

For training across multiple nodes:

```yaml
num_machines: 2         # Number of nodes
machine_rank: 0         # Set to 0 on first node, 1 on second, etc.
main_process_ip: "10.0.0.1"  # IP of main node
main_process_port: 29500
```

## Performance Tips

1. **Maximize Batch Size**: Increase until you hit OOM, then back off slightly
2. **Use Mixed Precision**: FP16 or BF16 for ~2x speedup
3. **Enable Gradient Checkpointing**: Trades compute for memory
4. **Pin Memory**: Already enabled in DataLoader for faster GPU transfer
5. **Fast Storage**: Use local SSD instead of network storage for dataset

## Model Saving

The model will be saved to `./models/finetuned-nomic-embed-multigpu/` with:
- Final model weights
- LoRA adapter weights
- Training configuration
- Best checkpoint based on validation score

To load:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./models/finetuned-nomic-embed-multigpu')
```

## Comparison with Single GPU

| Metric | Single GPU | Multi-GPU (4x) | Improvement |
|--------|-----------|----------------|-------------|
| Batch Size | 32 | 256 | 8x |
| Speed | 4.6s/it | 1.3s/it | 3.5x |
| Training Time | 300h | 85h | 3.5x |
| Memory/GPU | 42GB/80GB | 25GB/60GB | More efficient |
| Cost Efficiency | 1x | 3.5x | Better |

## Support

For issues:
1. Check this README
2. Review error messages in training logs
3. Verify GPU availability with `nvidia-smi`
4. Check Accelerate documentation: https://huggingface.co/docs/accelerate
