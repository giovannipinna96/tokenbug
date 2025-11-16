#!/bin/bash
#SBATCH --job-name=tokenbug_full_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=giovanni.pinna@phd.units.it

################################################################################
# TOKENBUG - FULL EXPERIMENTAL PIPELINE
################################################################################
# This script runs the complete pipeline for the bug detection paper:
# 1. Dataset creation from buggy snippets
# 2. Training with 3 loss functions (MNR, SupCon, Cosine)
# 3. Evaluation of all models
# 4. Bug detection framework tests
# 5. Results comparison and aggregation
################################################################################

set -e  # Exit on any error
set -u  # Exit on undefined variable

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################
# Usage:
#   sbatch run_full_pipeline.slurm                           # Run all steps
#   sbatch run_full_pipeline.slurm --skip-steps 1            # Skip dataset creation
#   sbatch run_full_pipeline.slurm --skip-steps 1,2          # Skip steps 1 and 2
#   sbatch run_full_pipeline.slurm --start-from-loss supcon  # Start training from supcon
#   sbatch run_full_pipeline.slurm --only-steps 2,3          # Run only steps 2 and 3
#   sbatch run_full_pipeline.slurm --losses supcon,cosine    # Train only these losses
################################################################################

# Default values
SKIP_STEPS=""
ONLY_STEPS=""
START_FROM_LOSS=""
SELECTED_LOSSES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-steps)
            SKIP_STEPS="$2"
            shift 2
            ;;
        --only-steps)
            ONLY_STEPS="$2"
            shift 2
            ;;
        --start-from-loss)
            START_FROM_LOSS="$2"
            shift 2
            ;;
        --losses)
            SELECTED_LOSSES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: sbatch run_full_pipeline.slurm [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-steps STEPS      Skip specific steps (comma-separated, e.g., 1,2)"
            echo "  --only-steps STEPS      Run only specific steps (comma-separated, e.g., 2,3)"
            echo "  --start-from-loss LOSS  Start training from specific loss (mnr, supcon, cosine)"
            echo "  --losses LOSSES         Train only specific losses (comma-separated)"
            echo ""
            echo "Steps:"
            echo "  1 - Dataset Creation"
            echo "  2 - Model Training"
            echo "  3 - Model Evaluation"
            echo "  4 - Bug Detection Tests"
            echo "  5 - Results Comparison"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper function to check if step should run
should_run_step() {
    local step=$1

    # If ONLY_STEPS is set, check if step is in the list
    if [ -n "${ONLY_STEPS}" ]; then
        if [[ ",${ONLY_STEPS}," == *",${step},"* ]]; then
            return 0  # Run this step
        else
            return 1  # Skip this step
        fi
    fi

    # If SKIP_STEPS is set, check if step should be skipped
    if [ -n "${SKIP_STEPS}" ]; then
        if [[ ",${SKIP_STEPS}," == *",${step},"* ]]; then
            return 1  # Skip this step
        fi
    fi

    return 0  # Run this step
}

################################################################################
# CONFIGURATION
################################################################################

# Directories
WORK_DIR="/u/gpinna/phd_projects/tokenbug/tokenbug"
LOG_DIR="${WORK_DIR}/logs"
RESULTS_DIR="${WORK_DIR}/paper_results"
DATA_DIR="${WORK_DIR}/data/hf_dataset"

# Training parameters
BATCH_SIZE=32
EPOCHS=1
LEARNING_RATE=3e-4
MAX_TRAIN_SAMPLES=850000
MAX_VAL_SAMPLES=780000

# Loss functions to compare
if [ -n "${SELECTED_LOSSES}" ]; then
    IFS=',' read -ra LOSS_FUNCTIONS <<< "${SELECTED_LOSSES}"
else
    LOSS_FUNCTIONS=("mnr" "supcon" "cosine")
fi

# Filter losses based on --start-from-loss
if [ -n "${START_FROM_LOSS}" ]; then
    FILTERED_LOSSES=()
    FOUND_START=false
    for loss in "${LOSS_FUNCTIONS[@]}"; do
        if [ "${loss}" == "${START_FROM_LOSS}" ]; then
            FOUND_START=true
        fi
        if [ "${FOUND_START}" == true ]; then
            FILTERED_LOSSES+=("${loss}")
        fi
    done
    if [ ${#FILTERED_LOSSES[@]} -eq 0 ]; then
        echo "ERROR: Loss function '${START_FROM_LOSS}' not found in list"
        exit 1
    fi
    LOSS_FUNCTIONS=("${FILTERED_LOSSES[@]}")
fi

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${WORK_DIR}/evaluation_results"

################################################################################
# SETUP
################################################################################

echo "################################################################################"
echo "# TOKENBUG FULL PIPELINE - Started at $(date)"
echo "################################################################################"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Working directory: ${WORK_DIR}"
echo ""

cd "${WORK_DIR}"

# Load modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.11

# Check GPU availability
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Check UV installation
echo "=== UV Version ==="
uv --version
echo ""

# Activate environment
echo "=== Python Environment ==="
uv sync
uv run python --version
echo ""

# Show configuration
echo "=== Run Configuration ==="
if [ -n "${SKIP_STEPS}" ]; then
    echo "Skipping steps: ${SKIP_STEPS}"
fi
if [ -n "${ONLY_STEPS}" ]; then
    echo "Running only steps: ${ONLY_STEPS}"
fi
if [ -n "${START_FROM_LOSS}" ]; then
    echo "Starting from loss: ${START_FROM_LOSS}"
fi
echo "Loss functions to train: ${LOSS_FUNCTIONS[*]}"
echo ""

# Initialize timing variables for skipped steps
STEP1_DURATION=0
STEP2_DURATION=0
STEP3_DURATION=0
STEP4_DURATION=0
STEP5_DURATION=0
PIPELINE_START=$(date +%s)

################################################################################
# STEP 1: DATASET CREATION
################################################################################

if should_run_step 1; then
    echo "################################################################################"
    echo "# STEP 1: Dataset Creation"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP1_START=$(date +%s)

    uv run python create_hf_dataset_optimized.py 2>&1 | tee "${LOG_DIR}/step1_dataset_creation_${SLURM_JOB_ID}.log"

    STEP1_END=$(date +%s)
    STEP1_DURATION=$((STEP1_END - STEP1_START))

    echo ""
    echo "✅ Step 1 completed in ${STEP1_DURATION} seconds"
    echo ""

    # Verify dataset was created
    if [ ! -d "${DATA_DIR}" ]; then
        echo "❌ ERROR: Dataset directory not found at ${DATA_DIR}"
        exit 1
    fi

    echo "Dataset created successfully at ${DATA_DIR}"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 1: Dataset Creation - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 2: MODEL TRAINING (ALL LOSS FUNCTIONS)
################################################################################

if should_run_step 2; then
    echo "################################################################################"
    echo "# STEP 2: Model Training"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP2_START=$(date +%s)

    for loss in "${LOSS_FUNCTIONS[@]}"; do
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "Training model with ${loss} loss function..."
    echo "Started at: $(date)"
    echo "────────────────────────────────────────────────────────────────────────────"
    echo ""

    TRAIN_START=$(date +%s)

    # Set temperature for SupCon
    if [ "${loss}" == "supcon" ]; then
        EXTRA_ARGS="--temperature 0.07"
    else
        EXTRA_ARGS=""
    fi

    # Run training
    uv run python train_with_contrastive.py \
        --loss-function "${loss}" \
        --model-name nomic-ai/nomic-embed-code \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --learning-rate ${LEARNING_RATE} \
        --use-lora \
        --lora-rank 16 \
        --max-train-samples ${MAX_TRAIN_SAMPLES} \
        --max-val-samples ${MAX_VAL_SAMPLES} \
        --warmup-steps 1000 \
        --evaluation-steps 2000 \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_DIR}/step2_train_${loss}_${SLURM_JOB_ID}.log"

    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

    echo ""
    echo "✅ Training with ${loss} loss completed in ${TRAIN_DURATION} seconds"
    echo ""

    # Verify model was saved
    MODEL_PATH="./models/finetuned-nomic-embed-code-${loss}"
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "⚠️  WARNING: Model not found at ${MODEL_PATH}"
    else
        echo "Model saved successfully at ${MODEL_PATH}"
    fi
    echo ""
    done

    STEP2_END=$(date +%s)
    STEP2_DURATION=$((STEP2_END - STEP2_START))

    echo ""
    echo "✅ Step 2 (all models) completed in ${STEP2_DURATION} seconds"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 2: Model Training - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 3: MODEL EVALUATION
################################################################################

if should_run_step 3; then
    echo "################################################################################"
    echo "# STEP 3: Model Evaluation"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP3_START=$(date +%s)

    for loss in "${LOSS_FUNCTIONS[@]}"; do
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "Evaluating model with ${loss} loss function..."
    echo "Started at: $(date)"
    echo "────────────────────────────────────────────────────────────────────────────"
    echo ""

    EVAL_START=$(date +%s)

    MODEL_PATH="./models/finetuned-nomic-embed-code-${loss}"
    OUTPUT_DIR="./evaluation_results/${loss}"

    # Check if model exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "⚠️  WARNING: Model not found at ${MODEL_PATH}, skipping evaluation"
        continue
    fi

    # Run evaluation
    uv run python evaluate_model.py \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size 128 \
        2>&1 | tee "${LOG_DIR}/step3_eval_${loss}_${SLURM_JOB_ID}.log"

    EVAL_END=$(date +%s)
    EVAL_DURATION=$((EVAL_END - EVAL_START))

    echo ""
    echo "✅ Evaluation of ${loss} model completed in ${EVAL_DURATION} seconds"
    echo ""

    # Verify evaluation results
    if [ ! -f "${OUTPUT_DIR}/metrics.json" ]; then
        echo "⚠️  WARNING: Metrics file not found at ${OUTPUT_DIR}/metrics.json"
    else
        echo "Evaluation results saved at ${OUTPUT_DIR}"
        echo "Metrics preview:"
        head -20 "${OUTPUT_DIR}/metrics.json"
    fi
    echo ""
    done

    STEP3_END=$(date +%s)
    STEP3_DURATION=$((STEP3_END - STEP3_START))

    echo ""
    echo "✅ Step 3 (all evaluations) completed in ${STEP3_DURATION} seconds"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 3: Model Evaluation - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 4: BUG DETECTION FRAMEWORK TESTS
################################################################################

if should_run_step 4; then
    echo "################################################################################"
    echo "# STEP 4: Bug Detection Framework Tests"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP4_START=$(date +%s)

    uv run python test_bug_detection.py 2>&1 | tee "${LOG_DIR}/step4_bug_detection_${SLURM_JOB_ID}.log"

    STEP4_END=$(date +%s)
    STEP4_DURATION=$((STEP4_END - STEP4_START))

    echo ""
    echo "✅ Step 4 completed in ${STEP4_DURATION} seconds"
    echo ""

    # Verify bug detection results
    if [ -f "bug_detection_results.csv" ]; then
        echo "Bug detection results saved at bug_detection_results.csv"
        echo "Results preview:"
        head -10 bug_detection_results.csv
    else
        echo "⚠️  WARNING: Bug detection results file not found"
    fi
    echo ""
else
    echo "################################################################################"
    echo "# STEP 4: Bug Detection Framework Tests - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 5: RESULTS COMPARISON AND AGGREGATION
################################################################################

if should_run_step 5; then
    echo "################################################################################"
    echo "# STEP 5: Results Comparison and Aggregation"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP5_START=$(date +%s)

    # Create comparison script on-the-fly
    cat > compare_models_temp.py << 'EOF'
"""
Compare all trained models and generate summary tables for the paper.
"""
import pandas as pd
import json
from pathlib import Path
import sys

def main():
    results = {}
    loss_types = ['mnr', 'supcon', 'cosine']

    print("=== Loading Evaluation Results ===\n")

    for loss_type in loss_types:
        eval_dir = Path(f'./evaluation_results/{loss_type}')
        metrics_file = eval_dir / 'metrics.json'

        if not metrics_file.exists():
            print(f"⚠️  WARNING: Metrics file not found for {loss_type}")
            continue

        # Load metrics
        with open(metrics_file) as f:
            metrics = json.load(f)

        results[loss_type] = {
            'Accuracy': metrics.get('accuracy', 0.0),
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'F1-Score': metrics.get('f1_score', 0.0),
            'ROC-AUC': metrics.get('roc_auc', 0.0),
            'PR-AUC': metrics.get('pr_auc', 0.0),
        }
        print(f"✅ Loaded results for {loss_type}")

    if not results:
        print("\n❌ ERROR: No evaluation results found!")
        sys.exit(1)

    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    df.index.name = 'Loss Function'

    # Save results
    output_dir = Path('./paper_results')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'model_comparison.csv'
    latex_path = output_dir / 'model_comparison.tex'

    df.to_csv(csv_path)
    df.to_latex(latex_path, float_format="%.4f")

    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(df.round(4).to_string())
    print("="*80)

    # Find best model
    best_f1_model = df['F1-Score'].idxmax()
    best_f1_score = df['F1-Score'].max()
    best_auc_model = df['ROC-AUC'].idxmax()
    best_auc_score = df['ROC-AUC'].max()

    print(f"\n🏆 Best F1-Score: {best_f1_model.upper()} ({best_f1_score:.4f})")
    print(f"🏆 Best ROC-AUC: {best_auc_model.upper()} ({best_auc_score:.4f})")

    print(f"\n✅ Results saved to:")
    print(f"   - {csv_path}")
    print(f"   - {latex_path}")

    # Save summary
    summary = {
        'best_f1_model': best_f1_model,
        'best_f1_score': float(best_f1_score),
        'best_auc_model': best_auc_model,
        'best_auc_score': float(best_auc_score),
        'all_results': df.to_dict()
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"   - {output_dir / 'summary.json'}")
    print()

if __name__ == '__main__':
    main()
EOF

    # Run comparison
    uv run python compare_models_temp.py 2>&1 | tee "${LOG_DIR}/step5_comparison_${SLURM_JOB_ID}.log"

    # Clean up temporary script
    rm -f compare_models_temp.py

    STEP5_END=$(date +%s)
    STEP5_DURATION=$((STEP5_END - STEP5_START))

    echo ""
    echo "✅ Step 5 completed in ${STEP5_DURATION} seconds"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 5: Results Comparison and Aggregation - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# FINAL SUMMARY
################################################################################

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - PIPELINE_START))

echo "################################################################################"
echo "# PIPELINE COMPLETED SUCCESSFULLY"
echo "# Finished at: $(date)"
echo "################################################################################"
echo ""
echo "⏱️  Timing Summary:"
echo "   Step 1 (Dataset):        ${STEP1_DURATION}s ($(($STEP1_DURATION / 60))m)"
echo "   Step 2 (Training):       ${STEP2_DURATION}s ($(($STEP2_DURATION / 60))m)"
echo "   Step 3 (Evaluation):     ${STEP3_DURATION}s ($(($STEP3_DURATION / 60))m)"
echo "   Step 4 (Bug Detection):  ${STEP4_DURATION}s ($(($STEP4_DURATION / 60))m)"
echo "   Step 5 (Comparison):     ${STEP5_DURATION}s ($(($STEP5_DURATION / 60))m)"
echo "   ────────────────────────────────────────"
echo "   TOTAL:                   ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m, $(($TOTAL_DURATION / 3600))h)"
echo ""

echo "📁 Results Location:"
echo "   Paper results:           ${RESULTS_DIR}/"
echo "   Evaluation results:      ${WORK_DIR}/evaluation_results/"
echo "   Bug detection results:   ${WORK_DIR}/bug_detection_results.csv"
echo "   Logs:                    ${LOG_DIR}/"
echo ""

echo "📊 Key Files for Paper:"
echo "   - ${RESULTS_DIR}/model_comparison.csv"
echo "   - ${RESULTS_DIR}/model_comparison.tex"
echo "   - ${RESULTS_DIR}/summary.json"
echo "   - ${WORK_DIR}/bug_detection_visualization.png"
echo ""

# List all generated files
echo "📦 All Generated Files:"
find "${RESULTS_DIR}" -type f 2>/dev/null | sort
find "./evaluation_results" -name "*.json" -o -name "*.png" -o -name "*.csv" 2>/dev/null | sort
echo ""

echo "✅ Pipeline completed successfully!"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

################################################################################
# OPTIONAL: Copy results to backup location
################################################################################

# Uncomment to backup results
# BACKUP_DIR="/path/to/backup/tokenbug_results_${SLURM_JOB_ID}"
# mkdir -p "${BACKUP_DIR}"
# cp -r "${RESULTS_DIR}" "${BACKUP_DIR}/"
# cp -r "./evaluation_results" "${BACKUP_DIR}/"
# cp bug_detection_results.csv "${BACKUP_DIR}/"
# echo "📦 Results backed up to ${BACKUP_DIR}"

exit 0
