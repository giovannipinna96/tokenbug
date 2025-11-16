# SQL Bug Detection Pipeline

Complete pipeline for training and evaluating SQL bug detection models using embedding similarity.

## Overview

This pipeline trains embedding models to detect buggy SQL lines by learning similarity patterns between SQL lines and their context. The approach is adapted from the Python bug detection framework in this repository.

## Pipeline Architecture

```
┌─────────────────────────────────┐
│ 1. generate_buggy_sql.py        │  Generate buggy SQL queries using LLM
│    Input: SQL example JSON      │
│    Output: Buggy queries JSON   │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│ 2. process_sql_bugs.py          │  Compute diffs and label buggy lines
│    Input: Buggy queries JSON    │
│    Output: Processed dataset    │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│ 3. create_sql_dataset.py        │  Create HuggingFace dataset with 3x context
│    Input: Processed dataset     │
│    Output: HF dataset           │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│ 4. train_sql_model.py           │  Train embedding model (4 loss options)
│    Input: HF dataset            │
│    Output: Trained model        │
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│ 5. evaluate_sql_model.py        │  Evaluate on NL2SQL-Bugs benchmark
│    Input: Trained model         │
│    Output: Metrics & results    │
└─────────────────────────────────┘
```

## Quick Start

### Step 1: Generate Buggy SQL Queries

```bash
# Create example input
cat > data/sql_example.json <<EOF
{
  "user_request": "Find all users older than 18 with verified emails",
  "table_name": "users",
  "database": "mydb",
  "columns": ["id", "name", "age", "email", "email_verified"],
  "correct_query": "SELECT * FROM users WHERE age > 18 AND email_verified = true"
}
EOF

# Generate 50 buggy queries
python generate_buggy_sql.py \
  --input-json data/sql_example.json \
  --output-json data/buggy_sql_output.json \
  --model-name microsoft/Phi-3-mini-4k-instruct \
  --num-buggy-queries 50 \
  --batch-size 8 \
  --temperature 0.8
```

**Output:** `data/buggy_sql_output.json` with 50 buggy SQL variations

---

### Step 2: Process Buggy Queries

```bash
# Process buggy queries and identify buggy lines via diff
python process_sql_bugs.py \
  --input-json data/buggy_sql_output.json \
  --output-json data/processed_sql_dataset.json \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

**What this does:**
- Computes diff between buggy and correct queries
- Identifies which lines are buggy (0-based indices)
- Adds correct queries as "stable" examples
- Splits into train/validation/test sets

**Output:** `data/processed_sql_dataset.json`

---

### Step 3: Create HuggingFace Dataset

```bash
# Create HF dataset with 3x context augmentation
python create_sql_dataset.py \
  --input-json data/processed_sql_dataset.json \
  --output-dir data/sql_hf_dataset \
  --context-size 3
```

**What this does:**
- Each SQL line generates 3 entries:
  - **before**: Current line + 3 preceding lines
  - **after**: Current line + 3 following lines
  - **full**: Current line + entire query
- Memory-efficient generator-based processing
- HuggingFace Dataset format for easy loading

**Output:** `data/sql_hf_dataset/` directory

**Dataset Structure:**
```python
{
  'current_line': str,      # Single SQL line
  'line_index': int,        # Position in query (0-based)
  'context': str,           # Surrounding code
  'context_type': str,      # 'before', 'after', or 'full'
  'score': int,             # 0 (buggy) or 1 (correct)
  'split': str,             # 'train', 'validation', 'test'
  'dataset_type': str,      # 'buggy' or 'stable'
  'query_type': str,        # 'SELECT', 'JOIN', etc.
  'error_type': str         # 'logical', 'syntactic', 'mixed'
}
```

---

### Step 4: Train Model

Train with one of 4 loss functions:

#### Option A: MNR Loss (Recommended)

```bash
python train_sql_model.py \
  --data-dir data/sql_hf_dataset \
  --model-name nomic-ai/nomic-embed-code \
  --loss-function mnr \
  --output-dir models/sql-bug-detector-mnr \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --use-lora
```

**MNR (MultipleNegativesRankingLoss):**
- Uses only positive pairs (correct SQL lines)
- State-of-the-art for embedding similarity
- Recommended for best performance

#### Option B: Cosine Similarity Loss

```bash
python train_sql_model.py \
  --data-dir data/sql_hf_dataset \
  --model-name nomic-ai/nomic-embed-code \
  --loss-function cosine \
  --output-dir models/sql-bug-detector-cosine \
  --use-weighted-sampling \
  --epochs 3
```

**Cosine Similarity:**
- Uses all data (buggy + correct) with labels
- Directly optimizes cosine similarity
- Good for interpretability

#### Option C: Supervised Contrastive Loss

```bash
python train_sql_model.py \
  --data-dir data/sql_hf_dataset \
  --model-name nomic-ai/nomic-embed-code \
  --loss-function supcon \
  --output-dir models/sql-bug-detector-supcon \
  --epochs 3
```

**SupCon:**
- Contrastive learning with supervision
- Pulls same-class embeddings together
- Pushes different-class embeddings apart

#### Option D: Ensemble Loss

```bash
python train_sql_model.py \
  --data-dir data/sql_hf_dataset \
  --model-name nomic-ai/nomic-embed-code \
  --loss-function ensemble \
  --ensemble-weights 0.4 0.3 0.3 \
  --ensemble-temperature 0.07 \
  --output-dir models/sql-bug-detector-ensemble \
  --epochs 3 \
  --use-lora
```

**Ensemble:**
- Combines all three losses with weights
- `--ensemble-weights`: [cosine, mnr, supcon] weights
- Most comprehensive but slower training

---

### Step 5: Evaluate on NL2SQL-Bugs Benchmark

```bash
# Download benchmark and evaluate
python evaluate_sql_model.py \
  --model-path models/sql-bug-detector-mnr \
  --benchmark-data data/nl2sql_bugs/NL2SQL-Bugs.json \
  --download-benchmark \
  --threshold 0.5 \
  --context-type full \
  --output-dir results/nl2sql_bugs_eval
```

**What this does:**
- Downloads NL2SQL-Bugs benchmark (2,018 examples)
- Computes line-by-line similarity scores
- Predicts buggy lines (similarity < threshold)
- Computes metrics:
  - Overall Accuracy
  - Precision, Recall, F1 (buggy class)
  - Type-Specific Accuracy per error category
  - Confusion Matrix

**Output:** `results/nl2sql_bugs_eval/results_threshold_0.5.json`

---

## Detailed Options

### generate_buggy_sql.py

```bash
--input-json PATH           # Input JSON with correct SQL
--output-json PATH          # Output JSON for buggy queries
--model-name MODEL          # HuggingFace model (default: Phi-3-mini)
--num-buggy-queries INT     # Number of buggy queries (default: 5)
--batch-size INT            # Batch size for generation (default: 8)
--temperature FLOAT         # Temperature for generation (default: 0.8)
--max-new-tokens INT        # Max tokens to generate (default: 512)
--device {cuda,cpu}         # Device (default: auto-detect)
```

### process_sql_bugs.py

```bash
--input-json PATH           # Buggy queries JSON
--output-json PATH          # Processed dataset JSON
--train-ratio FLOAT         # Train split ratio (default: 0.7)
--val-ratio FLOAT           # Validation split ratio (default: 0.15)
--test-ratio FLOAT          # Test split ratio (default: 0.15)
--no-stable                 # Don't include stable examples
```

### create_sql_dataset.py

```bash
--input-json PATH           # Processed dataset JSON
--output-dir PATH           # HF dataset output directory
--context-size INT          # Context lines before/after (default: 3)
--quiet                     # Suppress statistics output
```

### train_sql_model.py

```bash
# Data
--data-dir PATH             # HF dataset directory
--output-dir PATH           # Model output directory
--max-train-samples INT     # Limit training samples (for testing)

# Model
--model-name MODEL          # Base model (default: nomic-embed-code)
--loss-function {mnr,cosine,supcon,ensemble}  # Loss function

# Training
--epochs INT                # Number of epochs (default: 3)
--batch-size INT            # Batch size (default: 32)
--learning-rate FLOAT       # Learning rate (default: 2e-5)
--warmup-steps INT          # Warmup steps (default: 1000)
--eval-steps INT            # Evaluation interval (default: 2000)
--max-seq-length INT        # Max sequence length (default: 512)

# Optimization
--use-lora                  # Use LoRA for efficient fine-tuning
--lora-rank INT             # LoRA rank (default: 16)
--lora-alpha INT            # LoRA alpha (default: 32)
--lora-dropout FLOAT        # LoRA dropout (default: 0.1)

# Ensemble (if loss-function=ensemble)
--ensemble-weights F F F    # Weights for [cosine, mnr, supcon]
--ensemble-temperature F    # SupCon temperature (default: 0.07)

# Misc
--use-weighted-sampling     # Weighted sampling for class balance
--device {cuda,cpu}         # Device
--seed INT                  # Random seed (default: 42)
```

### evaluate_sql_model.py

```bash
--model-path PATH           # Trained model directory
--benchmark-data PATH       # NL2SQL-Bugs.json path
--download-benchmark        # Download benchmark if not found
--threshold FLOAT           # Similarity threshold (default: 0.5)
--context-type {before,after,full}  # Context type (default: full)
--output-dir PATH           # Results output directory
--device {cuda,cpu}         # Device
```

---

## Recommended Model Configurations

### Small/Fast (Experimentation)

```bash
# Model: distilroberta-base or codebert-base
# Batch size: 64
# Epochs: 3
# LoRA: enabled
# Training time: ~30 minutes on single GPU
```

### Medium (Production)

```bash
# Model: nomic-ai/nomic-embed-code
# Batch size: 32
# Epochs: 3
# LoRA: optional
# Training time: ~1-2 hours on single GPU
```

### Large (Best Performance)

```bash
# Model: deepseek-ai/deepseek-coder-6.7b-base
# Batch size: 16
# Epochs: 5
# LoRA: required
# Training time: ~4-6 hours on single GPU
```

---

## Loss Function Comparison

| Loss Function | Data Used | Best For | Speed | Performance |
|--------------|-----------|----------|-------|-------------|
| **MNR** | Positive pairs only | General purpose | Fast | ⭐⭐⭐⭐⭐ |
| **Cosine** | All data with labels | Interpretability | Fast | ⭐⭐⭐⭐ |
| **SupCon** | All data with contrastive | Class separation | Medium | ⭐⭐⭐⭐ |
| **Ensemble** | All data, combined | Maximum performance | Slow | ⭐⭐⭐⭐⭐ |

**Recommendation:** Start with MNR, then try Ensemble if you have compute budget.

---

## Expected Performance

Based on similar code bug detection tasks:

| Metric | Expected Range |
|--------|---------------|
| Overall Accuracy | 70-85% |
| Precision (buggy) | 60-75% |
| Recall (buggy) | 65-80% |
| F1-Score (buggy) | 65-77% |

**Factors affecting performance:**
- Quality and quantity of training data
- Model size and architecture
- Loss function choice
- Threshold tuning
- SQL complexity in test set

---

## Tips and Best Practices

### Data Generation

1. **Generate diverse bugs:** Use multiple LLMs or temperature settings
2. **Include both types:** Mix logical and syntactic errors
3. **Scale up:** Aim for 1000+ buggy queries for good coverage
4. **Balance dataset:** Include stable (correct) examples

### Training

1. **Start small:** Use `--max-train-samples 10000` for quick experiments
2. **Monitor overfitting:** Check validation metrics during training
3. **Use LoRA:** Enables larger batch sizes and faster training
4. **Try different models:** Code-specific models often work best

### Evaluation

1. **Tune threshold:** Try multiple thresholds (0.3, 0.5, 0.7)
2. **Analyze errors:** Look at false positives/negatives
3. **Per-type metrics:** Check performance on specific error types
4. **Context experiments:** Test before/after/full context types

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 8

# Enable LoRA
--use-lora

# Use smaller model
--model-name microsoft/codebert-base

# Limit training samples
--max-train-samples 50000
```

### Low Performance

```bash
# Generate more training data
--num-buggy-queries 1000

# Use weighted sampling
--use-weighted-sampling

# Try ensemble loss
--loss-function ensemble

# Increase epochs
--epochs 5

# Try different threshold
--threshold 0.3  # More sensitive
--threshold 0.7  # More conservative
```

### Slow Training

```bash
# Increase batch size (if memory allows)
--batch-size 64

# Reduce evaluation frequency
--eval-steps 5000

# Use faster model
--model-name distilroberta-base
```

---

## File Structure

```
tokenbug/
├── generate_buggy_sql.py          # Step 1: Generate buggy SQL
├── process_sql_bugs.py            # Step 2: Process and label
├── create_sql_dataset.py          # Step 3: Create HF dataset
├── train_sql_model.py             # Step 4: Train model
├── evaluate_sql_model.py          # Step 5: Evaluate on benchmark
├── sql_utils.py                   # Utility functions
├── training_utils.py              # Training helpers (shared with Python)
├── data/
│   ├── sql_example.json           # Example input
│   ├── buggy_sql_output.json      # Generated buggy queries
│   ├── processed_sql_dataset.json # Processed dataset
│   ├── sql_hf_dataset/            # HuggingFace dataset
│   └── nl2sql_bugs/               # NL2SQL-Bugs benchmark
├── models/
│   └── sql-bug-detector-*/        # Trained models
└── results/
    └── nl2sql_bugs_eval/          # Evaluation results
```

---

## Citation

If you use this SQL bug detection pipeline, please cite:

```bibtex
@misc{sql-bug-detection-2025,
  title={SQL Bug Detection via Embedding Similarity},
  author={Your Name},
  year={2025},
  note={Based on tokenbug framework}
}
```

**NL2SQL-Bugs Benchmark:**
```bibtex
@inproceedings{nl2sql-bugs,
  title={NL2SQL-Bugs: A Benchmark for Detecting Semantic Errors in NL-to-SQL Translation},
  author={HKUSTDial},
  year={2024},
  url={https://github.com/HKUSTDial/NL2SQL-Bugs-Benchmark}
}
```

---

## License

Same license as the parent tokenbug repository.

---

## Support

For issues or questions:
1. Check this README and code comments
2. Review the Python bug detection pipeline (similar architecture)
3. Open an issue on the repository

---

**Happy SQL Bug Hunting! 🐛🔍**
