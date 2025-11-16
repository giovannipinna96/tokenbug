#!/bin/bash
#SBATCH --job-name=sql_bug_pipeline
#SBATCH --output=logs/sql_pipeline_%j.out
#SBATCH --error=logs/sql_pipeline_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=giovanni.pinna@phd.units.it

################################################################################
# SQL BUG DETECTION - FULL EXPERIMENTAL PIPELINE
################################################################################
# This script runs the complete SQL bug detection pipeline:
# 1. Generate buggy SQL queries using LLM
# 2. Process queries and identify buggy lines via diff
# 3. Create HuggingFace dataset with 3x context augmentation
# 4. Train models with 4 loss functions (MNR, Cosine, SupCon, Ensemble)
# 5. Evaluate all models on NL2SQL-Bugs benchmark
# 6. Compare results and generate summary tables
################################################################################

set -e  # Exit on any error
set -u  # Exit on undefined variable

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################
# Usage:
#   sbatch run_sql_pipeline.slurm                           # Run all steps
#   sbatch run_sql_pipeline.slurm --skip-steps 1,2,3       # Skip steps 1,2,3
#   sbatch run_sql_pipeline.slurm --start-from-loss supcon  # Start training from supcon
#   sbatch run_sql_pipeline.slurm --only-steps 4,5          # Run only steps 4 and 5
#   sbatch run_sql_pipeline.slurm --losses supcon,ensemble  # Train only these losses
################################################################################

# Default values
SKIP_STEPS=""
ONLY_STEPS=""
START_FROM_LOSS=""
SELECTED_LOSSES=""
FULL_CONTEXT_ONLY=false

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
        --full-context-only)
            FULL_CONTEXT_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: sbatch run_sql_pipeline.slurm [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-steps STEPS      Skip specific steps (comma-separated, e.g., 1,2,3)"
            echo "  --only-steps STEPS      Run only specific steps (comma-separated, e.g., 4,5)"
            echo "  --start-from-loss LOSS  Start training from specific loss (mnr, cosine, supcon, ensemble)"
            echo "  --losses LOSSES         Train only specific losses (comma-separated)"
            echo "  --full-context-only     Use only full SQL query as context (no 3x augmentation)"
            echo ""
            echo "Steps:"
            echo "  0 - Check/Create Input Data"
            echo "  1 - Generate Buggy SQL Queries"
            echo "  2 - Process SQL Bugs (Diff and Labeling)"
            echo "  3 - Create HuggingFace Dataset"
            echo "  4 - Model Training"
            echo "  5 - Model Evaluation"
            echo "  6 - Results Comparison"
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
RESULTS_DIR="${WORK_DIR}/sql_results"
DATA_DIR="${WORK_DIR}/data"
MODELS_DIR="${WORK_DIR}/models"
EVAL_DIR="${WORK_DIR}/sql_evaluation_results"

# Input/Output files
SQL_INPUT_JSON="${DATA_DIR}/sql_example.json"
BUGGY_SQL_OUTPUT="${DATA_DIR}/buggy_sql_output.json"
PROCESSED_SQL_JSON="${DATA_DIR}/processed_sql_dataset.json"
SQL_HF_DATASET="${DATA_DIR}/sql_hf_dataset"
NL2SQL_BUGS_DATA="${DATA_DIR}/nl2sql_bugs/NL2SQL-Bugs.json"

# Generation parameters
NUM_BUGGY_QUERIES=500
LLM_MODEL="microsoft/Phi-3-mini-4k-instruct"
GEN_BATCH_SIZE=16
TEMPERATURE=0.8

# Training parameters
EMBEDDING_MODEL="nomic-ai/nomic-embed-code"
BATCH_SIZE=32
EPOCHS=3
LEARNING_RATE=2e-5
WARMUP_STEPS=1000
EVAL_STEPS=2000
MAX_SEQ_LENGTH=512
CONTEXT_SIZE=3

# Loss functions to compare (all 4)
if [ -n "${SELECTED_LOSSES}" ]; then
    IFS=',' read -ra LOSS_FUNCTIONS <<< "${SELECTED_LOSSES}"
else
    LOSS_FUNCTIONS=("mnr" "cosine" "supcon" "ensemble")
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

# Ensemble weights (for ensemble loss)
ENSEMBLE_WEIGHTS="0.4 0.3 0.3"  # [cosine, mnr, supcon]
ENSEMBLE_TEMP=0.07

# Evaluation parameters
EVAL_THRESHOLD=0.5
CONTEXT_TYPE="full"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${EVAL_DIR}"
mkdir -p "${DATA_DIR}"

################################################################################
# SETUP
################################################################################

echo "################################################################################"
echo "# SQL BUG DETECTION PIPELINE - Started at $(date)"
echo "################################################################################"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Working directory: ${WORK_DIR}"
echo ""

cd "${WORK_DIR}"

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
if [ "${FULL_CONTEXT_ONLY}" == true ]; then
    echo "Dataset mode: Full context only (no 3x augmentation)"
else
    echo "Dataset mode: 3x context augmentation (context_size=${CONTEXT_SIZE})"
fi
echo "Loss functions to train: ${LOSS_FUNCTIONS[*]}"
echo ""

# Initialize timing variables for skipped steps
STEP1_DURATION=0
STEP2_DURATION=0
STEP3_DURATION=0
STEP4_DURATION=0
STEP5_DURATION=0
STEP6_DURATION=0
PIPELINE_START=$(date +%s)

################################################################################
# STEP 0: CHECK/CREATE INPUT DATA
################################################################################

if should_run_step 0; then
    echo "################################################################################"
    echo "# STEP 0: Check/Create Input Data"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    # Check if input JSON exists, create example if not
    if [ ! -f "${SQL_INPUT_JSON}" ]; then
    echo "Creating example SQL input JSON..."
    mkdir -p "${DATA_DIR}"
    cat > "${SQL_INPUT_JSON}" <<EOF
{
  "user_request": "Find all users who are older than 18 years and have verified email addresses",
  "table_name": "users",
  "database": "mydb",
  "columns": ["id", "name", "age", "email", "email_verified", "created_at", "updated_at"],
  "correct_query": "SELECT * FROM users WHERE age > 18 AND email_verified = true ORDER BY created_at DESC"
}
EOF
    echo "✅ Created example input at ${SQL_INPUT_JSON}"
else
    echo "✅ Input JSON found at ${SQL_INPUT_JSON}"
    fi
    echo ""
else
    echo "################################################################################"
    echo "# STEP 0: Check/Create Input Data - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 1: GENERATE BUGGY SQL QUERIES
################################################################################

if should_run_step 1; then
    echo "################################################################################"
    echo "# STEP 1: Generate Buggy SQL Queries"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP1_START=$(date +%s)

    # Check if buggy queries already exist
    if [ -f "${BUGGY_SQL_OUTPUT}" ]; then
    echo "⚠️  Buggy SQL queries already exist at ${BUGGY_SQL_OUTPUT}"
    echo "Skipping generation. Delete file to regenerate."
    STEP1_DURATION=0
else
    echo "Generating ${NUM_BUGGY_QUERIES} buggy SQL queries..."
    echo "LLM Model: ${LLM_MODEL}"
    echo "Batch Size: ${GEN_BATCH_SIZE}"
    echo "Temperature: ${TEMPERATURE}"
    echo ""

    uv run python generate_buggy_sql.py \
        --input-json "${SQL_INPUT_JSON}" \
        --output-json "${BUGGY_SQL_OUTPUT}" \
        --model-name "${LLM_MODEL}" \
        --num-buggy-queries ${NUM_BUGGY_QUERIES} \
        --batch-size ${GEN_BATCH_SIZE} \
        --temperature ${TEMPERATURE} \
        --max-new-tokens 512 \
        2>&1 | tee "${LOG_DIR}/step1_generate_buggy_${SLURM_JOB_ID}.log"

    STEP1_END=$(date +%s)
    STEP1_DURATION=$((STEP1_END - STEP1_START))

    echo ""
    echo "✅ Step 1 completed in ${STEP1_DURATION} seconds ($(($STEP1_DURATION / 60))m)"
fi

# Verify output
if [ ! -f "${BUGGY_SQL_OUTPUT}" ]; then
    echo "❌ ERROR: Buggy SQL output not found at ${BUGGY_SQL_OUTPUT}"
    exit 1
fi

    echo "Buggy SQL queries available at ${BUGGY_SQL_OUTPUT}"
    echo "Number of entries: $(jq 'length' ${BUGGY_SQL_OUTPUT})"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 1: Generate Buggy SQL Queries - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 2: PROCESS SQL BUGS (DIFF AND LABELING)
################################################################################

if should_run_step 2; then
    echo "################################################################################"
    echo "# STEP 2: Process SQL Bugs (Diff and Labeling)"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP2_START=$(date +%s)

echo "Processing buggy SQL queries and identifying buggy lines..."
echo ""

uv run python process_sql_bugs.py \
    --input-json "${BUGGY_SQL_OUTPUT}" \
    --output-json "${PROCESSED_SQL_JSON}" \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    2>&1 | tee "${LOG_DIR}/step2_process_bugs_${SLURM_JOB_ID}.log"

STEP2_END=$(date +%s)
STEP2_DURATION=$((STEP2_END - STEP2_START))

echo ""
echo "✅ Step 2 completed in ${STEP2_DURATION} seconds ($(($STEP2_DURATION / 60))m)"
echo ""

# Verify output
if [ ! -f "${PROCESSED_SQL_JSON}" ]; then
    echo "❌ ERROR: Processed SQL dataset not found at ${PROCESSED_SQL_JSON}"
    exit 1
fi

    echo "Processed dataset saved at ${PROCESSED_SQL_JSON}"
    echo "Number of entries: $(jq 'length' ${PROCESSED_SQL_JSON})"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 2: Process SQL Bugs - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 3: CREATE HUGGINGFACE DATASET
################################################################################

if should_run_step 3; then
    echo "################################################################################"
    echo "# STEP 3: Create HuggingFace Dataset"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP3_START=$(date +%s)

if [ "${FULL_CONTEXT_ONLY}" == true ]; then
    echo "Creating HuggingFace dataset with full context only (no augmentation)..."
else
    echo "Creating HuggingFace dataset with 3x context augmentation..."
    echo "Context size: ${CONTEXT_SIZE}"
fi
echo ""

# Build dataset creation command
DATASET_CMD="uv run python create_sql_dataset.py \
    --input-json ${PROCESSED_SQL_JSON} \
    --output-dir ${SQL_HF_DATASET} \
    --context-size ${CONTEXT_SIZE}"

if [ "${FULL_CONTEXT_ONLY}" == true ]; then
    DATASET_CMD="${DATASET_CMD} --full-context-only"
fi

eval ${DATASET_CMD} 2>&1 | tee "${LOG_DIR}/step3_create_dataset_${SLURM_JOB_ID}.log"

STEP3_END=$(date +%s)
STEP3_DURATION=$((STEP3_END - STEP3_START))

echo ""
echo "✅ Step 3 completed in ${STEP3_DURATION} seconds ($(($STEP3_DURATION / 60))m)"
echo ""

# Verify dataset
if [ ! -d "${SQL_HF_DATASET}" ]; then
    echo "❌ ERROR: HuggingFace dataset not found at ${SQL_HF_DATASET}"
    exit 1
fi

    echo "HuggingFace dataset created at ${SQL_HF_DATASET}"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 3: Create HuggingFace Dataset - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 4: MODEL TRAINING (ALL LOSS FUNCTIONS)
################################################################################

if should_run_step 4; then
    echo "################################################################################"
    echo "# STEP 4: Model Training (All Loss Functions)"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP4_START=$(date +%s)

    for loss in "${LOSS_FUNCTIONS[@]}"; do
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "Training model with ${loss^^} loss function..."
    echo "Started at: $(date)"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""

    TRAIN_START=$(date +%s)

    MODEL_OUTPUT_DIR="${MODELS_DIR}/sql-bug-detector-${loss}"

    # Build training command
    TRAIN_CMD="uv run python train_sql_model.py \
        --data-dir ${SQL_HF_DATASET} \
        --output-dir ${MODEL_OUTPUT_DIR} \
        --model-name ${EMBEDDING_MODEL} \
        --loss-function ${loss} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LEARNING_RATE} \
        --warmup-steps ${WARMUP_STEPS} \
        --eval-steps ${EVAL_STEPS} \
        --max-seq-length ${MAX_SEQ_LENGTH} \
        --use-lora \
        --lora-rank 16 \
        --lora-alpha 32 \
        --lora-dropout 0.1"

    # Add ensemble-specific parameters
    if [ "${loss}" == "ensemble" ]; then
        TRAIN_CMD="${TRAIN_CMD} \
            --ensemble-weights ${ENSEMBLE_WEIGHTS} \
            --ensemble-temperature ${ENSEMBLE_TEMP}"
    fi

    # Add weighted sampling for cosine loss
    if [ "${loss}" == "cosine" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use-weighted-sampling"
    fi

    # Execute training
    eval ${TRAIN_CMD} 2>&1 | tee "${LOG_DIR}/step4_train_${loss}_${SLURM_JOB_ID}.log"

    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

    echo ""
    echo "✅ Training with ${loss^^} loss completed in ${TRAIN_DURATION} seconds ($(($TRAIN_DURATION / 60))m)"
    echo ""

    # Verify model was saved
    if [ ! -d "${MODEL_OUTPUT_DIR}" ]; then
        echo "⚠️  WARNING: Model not found at ${MODEL_OUTPUT_DIR}"
    else
        echo "Model saved successfully at ${MODEL_OUTPUT_DIR}"
        # Show model size
        MODEL_SIZE=$(du -sh "${MODEL_OUTPUT_DIR}" | cut -f1)
        echo "Model size: ${MODEL_SIZE}"
    fi
    echo ""
    done

    STEP4_END=$(date +%s)
    STEP4_DURATION=$((STEP4_END - STEP4_START))

    echo ""
    echo "✅ Step 4 (all models) completed in ${STEP4_DURATION} seconds ($(($STEP4_DURATION / 60))m, $(($STEP4_DURATION / 3600))h)"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 4: Model Training - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 5: MODEL EVALUATION ON NL2SQL-BUGS BENCHMARK
################################################################################

if should_run_step 5; then
    echo "################################################################################"
    echo "# STEP 5: Model Evaluation on NL2SQL-Bugs Benchmark"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP5_START=$(date +%s)

    for loss in "${LOSS_FUNCTIONS[@]}"; do
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "Evaluating ${loss^^} model on NL2SQL-Bugs benchmark..."
    echo "Started at: $(date)"
    echo "────────────────────────────────────────────────────────────────────────────"
    echo ""

    EVAL_START=$(date +%s)

    MODEL_PATH="${MODELS_DIR}/sql-bug-detector-${loss}"
    OUTPUT_DIR="${EVAL_DIR}/${loss}"

    # Check if model exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "⚠️  WARNING: Model not found at ${MODEL_PATH}, skipping evaluation"
        continue
    fi

    # Run evaluation (will download benchmark if needed)
    uv run python evaluate_sql_model.py \
        --model-path "${MODEL_PATH}" \
        --benchmark-data "${NL2SQL_BUGS_DATA}" \
        --download-benchmark \
        --threshold ${EVAL_THRESHOLD} \
        --context-type ${CONTEXT_TYPE} \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/step5_eval_${loss}_${SLURM_JOB_ID}.log"

    EVAL_END=$(date +%s)
    EVAL_DURATION=$((EVAL_END - EVAL_START))

    echo ""
    echo "✅ Evaluation of ${loss^^} model completed in ${EVAL_DURATION} seconds ($(($EVAL_DURATION / 60))m)"
    echo ""

    # Verify evaluation results
    RESULTS_FILE="${OUTPUT_DIR}/results_threshold_${EVAL_THRESHOLD}.json"
    if [ ! -f "${RESULTS_FILE}" ]; then
        echo "⚠️  WARNING: Results file not found at ${RESULTS_FILE}"
    else
        echo "Evaluation results saved at ${RESULTS_FILE}"
        echo "Results preview:"
        jq '.overall_accuracy, .precision, .recall, .f1_score' "${RESULTS_FILE}"
    fi
    echo ""
    done

    STEP5_END=$(date +%s)
    STEP5_DURATION=$((STEP5_END - STEP5_START))

    echo ""
    echo "✅ Step 5 (all evaluations) completed in ${STEP5_DURATION} seconds ($(($STEP5_DURATION / 60))m)"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 5: Model Evaluation - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# STEP 6: RESULTS COMPARISON AND AGGREGATION
################################################################################

if should_run_step 6; then
    echo "################################################################################"
    echo "# STEP 6: Results Comparison and Aggregation"
    echo "# Started at: $(date)"
    echo "################################################################################"
    echo ""

    STEP6_START=$(date +%s)

    # Create comparison script on-the-fly
    cat > compare_sql_models_temp.py << 'EOF'
"""
Compare all trained SQL bug detection models and generate summary tables.
"""
import pandas as pd
import json
from pathlib import Path
import sys

def main():
    results = {}
    loss_types = ['mnr', 'cosine', 'supcon', 'ensemble']
    threshold = 0.5

    print("=== Loading SQL Bug Detection Evaluation Results ===\n")

    for loss_type in loss_types:
        eval_dir = Path(f'./sql_evaluation_results/{loss_type}')
        results_file = eval_dir / f'results_threshold_{threshold}.json'

        if not results_file.exists():
            print(f"⚠️  WARNING: Results file not found for {loss_type}")
            continue

        # Load results
        with open(results_file) as f:
            res = json.load(f)

        results[loss_type] = {
            'Overall Accuracy': res.get('overall_accuracy', 0.0),
            'Precision': res.get('precision', 0.0),
            'Recall': res.get('recall', 0.0),
            'F1-Score': res.get('f1_score', 0.0),
            'Num Samples': res.get('num_samples', 0),
            'Correct Predictions': res.get('num_correct_predictions', 0),
        }
        print(f"✅ Loaded results for {loss_type.upper()}")

    if not results:
        print("\n❌ ERROR: No evaluation results found!")
        sys.exit(1)

    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    df.index.name = 'Loss Function'

    # Save results
    output_dir = Path('./sql_results')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'sql_model_comparison.csv'
    latex_path = output_dir / 'sql_model_comparison.tex'
    json_path = output_dir / 'sql_model_comparison.json'

    df.to_csv(csv_path)
    df.to_latex(latex_path, float_format="%.4f")

    print("\n" + "="*80)
    print("SQL BUG DETECTION MODEL COMPARISON")
    print("="*80)
    print(df.round(4).to_string())
    print("="*80)

    # Find best models
    best_f1_model = df['F1-Score'].idxmax()
    best_f1_score = df['F1-Score'].max()
    best_acc_model = df['Overall Accuracy'].idxmax()
    best_acc_score = df['Overall Accuracy'].max()
    best_precision_model = df['Precision'].idxmax()
    best_precision_score = df['Precision'].max()
    best_recall_model = df['Recall'].idxmax()
    best_recall_score = df['Recall'].max()

    print(f"\n🏆 Best Overall Accuracy: {best_acc_model.upper()} ({best_acc_score:.4f})")
    print(f"🏆 Best Precision: {best_precision_model.upper()} ({best_precision_score:.4f})")
    print(f"🏆 Best Recall: {best_recall_model.upper()} ({best_recall_score:.4f})")
    print(f"🏆 Best F1-Score: {best_f1_model.upper()} ({best_f1_score:.4f})")

    print(f"\n✅ Results saved to:")
    print(f"   - {csv_path}")
    print(f"   - {latex_path}")

    # Save summary
    summary = {
        'threshold': threshold,
        'best_accuracy_model': best_acc_model,
        'best_accuracy_score': float(best_acc_score),
        'best_f1_model': best_f1_model,
        'best_f1_score': float(best_f1_score),
        'best_precision_model': best_precision_model,
        'best_precision_score': float(best_precision_score),
        'best_recall_model': best_recall_model,
        'best_recall_score': float(best_recall_score),
        'all_results': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv
                           for kk, vv in v.items()}
                       for k, v in results.items()}
    }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"   - {json_path}")
    print()

if __name__ == '__main__':
    main()
EOF

    # Run comparison
    uv run python compare_sql_models_temp.py 2>&1 | tee "${LOG_DIR}/step6_comparison_${SLURM_JOB_ID}.log"

    # Clean up temporary script
    rm -f compare_sql_models_temp.py

    STEP6_END=$(date +%s)
    STEP6_DURATION=$((STEP6_END - STEP6_START))

    echo ""
    echo "✅ Step 6 completed in ${STEP6_DURATION} seconds"
    echo ""
else
    echo "################################################################################"
    echo "# STEP 6: Results Comparison - SKIPPED"
    echo "################################################################################"
    echo ""
fi

################################################################################
# FINAL SUMMARY
################################################################################

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - PIPELINE_START))

echo "################################################################################"
echo "# SQL BUG DETECTION PIPELINE COMPLETED SUCCESSFULLY"
echo "# Finished at: $(date)"
echo "################################################################################"
echo ""
echo "⏱️  Timing Summary:"
echo "   Step 1 (Generate Buggy SQL):    ${STEP1_DURATION}s ($(($STEP1_DURATION / 60))m)"
echo "   Step 2 (Process & Label):       ${STEP2_DURATION}s ($(($STEP2_DURATION / 60))m)"
echo "   Step 3 (Create HF Dataset):     ${STEP3_DURATION}s ($(($STEP3_DURATION / 60))m)"
echo "   Step 4 (Training All Models):   ${STEP4_DURATION}s ($(($STEP4_DURATION / 60))m, $(($STEP4_DURATION / 3600))h)"
echo "   Step 5 (Evaluation):            ${STEP5_DURATION}s ($(($STEP5_DURATION / 60))m)"
echo "   Step 6 (Comparison):            ${STEP6_DURATION}s ($(($STEP6_DURATION / 60))m)"
echo "   ────────────────────────────────────────────────────────"
echo "   TOTAL:                          ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m, $(($TOTAL_DURATION / 3600))h)"
echo ""

echo "📁 Results Location:"
echo "   SQL Results:                ${RESULTS_DIR}/"
echo "   Evaluation Results:         ${EVAL_DIR}/"
echo "   Trained Models:             ${MODELS_DIR}/"
echo "   Logs:                       ${LOG_DIR}/"
echo ""

echo "📊 Key Files for Analysis:"
echo "   - ${RESULTS_DIR}/sql_model_comparison.csv"
echo "   - ${RESULTS_DIR}/sql_model_comparison.tex"
echo "   - ${RESULTS_DIR}/sql_model_comparison.json"
echo ""

# List all trained models
echo "🤖 Trained Models:"
ls -lh "${MODELS_DIR}"/sql-bug-detector-* 2>/dev/null | grep "^d" || echo "   No models found"
echo ""

# List all evaluation results
echo "📈 Evaluation Results:"
find "${EVAL_DIR}" -name "results_*.json" 2>/dev/null | sort || echo "   No results found"
echo ""

# Show comparison table
if [ -f "${RESULTS_DIR}/sql_model_comparison.csv" ]; then
    echo "📊 Model Comparison Summary:"
    cat "${RESULTS_DIR}/sql_model_comparison.csv"
    echo ""
fi

echo "✅ SQL Bug Detection Pipeline completed successfully!"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

################################################################################
# OPTIONAL: Copy results to backup location
################################################################################

# Uncomment to backup results
# BACKUP_DIR="/path/to/backup/sql_bug_results_${SLURM_JOB_ID}"
# mkdir -p "${BACKUP_DIR}"
# cp -r "${RESULTS_DIR}" "${BACKUP_DIR}/"
# cp -r "${EVAL_DIR}" "${BACKUP_DIR}/"
# cp -r "${MODELS_DIR}"/sql-bug-detector-* "${BACKUP_DIR}/"
# echo "📦 Results backed up to ${BACKUP_DIR}"

exit 0
