# Uncertainty-Based Bug Detection Framework

## Project Overview and Motivation

### Purpose

This project implements an **Uncertainty-Based Bug Detection Framework** that identifies potentially buggy lines in Python code and SQL queries using pre-trained language models. The system leverages multiple uncertainty quantification methods to detect anomalous code segments that likely contain bugs, whether logical or syntactic in nature.

### Why This Matters

Bug detection in source code is a fundamental challenge in software engineering. Understanding **where** bugs are located within a codebase is a critical first step toward correcting them. This task is exceptionally difficult because it requires:

1. **Technical Understanding**: Deep comprehension of the programming language itself, including syntax rules, semantic conventions, and language-specific idioms. The system must understand not only what makes code syntactically valid but also what patterns are typical for correct implementations.

2. **Intent Comprehension**: Understanding what the user or developer intended to achieve. A bug is not just an error in isolation—it represents a deviation from the programmer's intended behavior. This requires interpreting the context, the function's purpose, and the expected outcomes.

3. **Multi-dimensional Analysis**: Bugs can manifest in various forms:
   - **Syntactic errors**: Misspellings, missing punctuation, incorrect keywords
   - **Logical errors**: Wrong comparison operators (>= instead of >), incorrect conditions, off-by-one errors
   - **Semantic errors**: Using wrong column names in SQL, incorrect table joins, missing clauses

The framework addresses these challenges by employing multiple uncertainty quantification methods that analyze code from different perspectives, providing a robust ensemble approach to bug localization.

### Practical Applications

This bug detection system serves as a foundation for:
- **Automated Code Review**: Identifying suspicious code sections before deployment
- **LLM-Guided Bug Fixing**: Directing large language models to focus on specific problematic lines rather than entire codebases
- **SQL Query Debugging**: Understanding whether query errors stem from incorrect filtering, non-existent column names, or invalid aliases
- **Genetic Improvement**: Guiding evolutionary algorithms to prioritize mutation and crossover operations on likely buggy code segments

---

## Datasets

### Python Bug Detection Dataset

The training data for Python bug detection is derived from the **PyTraceBugs** dataset and custom-collected real bug fixes from open-source repositories.

#### Data Source Structure

The data consists of pairs of Python files:
- `{hash}_before_merge.py`: Code containing bugs
- `{hash}_after_merge.py`: Fixed version of the same code

These pairs represent actual bug fixes committed to version control systems, providing real-world examples of buggy and correct code.

#### Bug Line Identification via Diff

The system uses **difflib** to compute unified diffs between buggy and correct versions:

```python
diff = difflib.unified_diff(
    buggy_lines,
    correct_lines,
    lineterm='',
    n=0  # No context lines
)
```

Lines marked with `-` in the diff (present in buggy version but modified in the fix) are identified as buggy lines with their 0-based indices recorded.

#### Data Augmentation Strategy (3x Context Augmentation)

For each line of code in the dataset, the system generates **three training examples** with different context windows:

1. **context_before**: Current line + up to 3 preceding lines
   ```python
   def get_context_before(lines, line_idx, context_size=3):
       start_idx = max(0, line_idx - context_size)
       return '\n'.join(lines[start_idx:line_idx + 1])
   ```

2. **context_after**: Current line + up to 3 following lines
   ```python
   def get_context_after(lines, line_idx, context_size=3):
       end_idx = min(len(lines), line_idx + context_size + 1)
       return '\n'.join(lines[line_idx:end_idx])
   ```

3. **context_full**: Current line + entire function/code block
   ```python
   def get_context_full(lines, line_idx):
       return '\n'.join(lines)
   ```

This **3x augmentation** strategy:
- Increases training data volume significantly
- Teaches the model to recognize bugs in different contextual settings
- Improves generalization by exposing the model to various context windows
- Creates a label structure: **0 = buggy line**, **1 = correct line**

#### Dataset Statistics

- **Format**: HuggingFace Dataset (Arrow format)
- **Splits**: Train (70%), Validation (15%), Test (15%)
- **Features per entry**:
  - `current_line`: The line being analyzed
  - `context`: Surrounding context (before/after/full)
  - `context_type`: Type of context window
  - `score`: Binary label (0=buggy, 1=correct)
  - Metadata: function name, filename, traceback type

---

### SQL Bug Detection Dataset

#### Evaluation Benchmark: NL2SQL-Bugs

For SQL bug detection evaluation, the project uses the **NL2SQL-Bugs Benchmark**:
- 2,018 examples (999 incorrect, 1,019 correct SQL queries)
- 9 main error categories with 31 subcategories
- Covers semantic errors in Natural Language to SQL translation

This benchmark provides ground truth for evaluating the model's ability to detect bugs in SQL queries.

#### Training Data Generation

Since NL2SQL-Bugs is used for evaluation, training data is generated from:

1. **BIRD Dataset**: Business Intelligence Research Datasets
2. **Spider Dataset**: Cross-domain text-to-SQL benchmark

For each correct query from these datasets (excluding those in NL2SQL-Bugs):

**Step 1: Buggy Query Generation via LLM**

```python
class SQLBuggyQueryGenerator:
    ERROR_PROMPTS = {
        "logical_comparison": "Generate incorrect SQL by using wrong comparison operator",
        "logical_join": "Generate incorrect SQL by using wrong JOIN type",
        "logical_aggregation": "Generate incorrect SQL with wrong aggregation",
        "syntactic_table": "Generate incorrect SQL with table name syntax error",
        "syntactic_keyword": "Generate incorrect SQL with keyword typos",
        # ... more error types
    }
```

The system uses an LLM (e.g., `microsoft/Phi-3-mini-4k-instruct`) to generate **10 buggy variations** of each correct query, cycling through different error types to ensure variety.

**Step 2: Diff-Based Bug Line Identification**

Same as Python processing:
```python
buggy_lines_indices = infer_buggy_sql_lines_from_diff(
    buggy_query,
    correct_query,
    normalize=True
)
```

**Step 3: 3x Context Augmentation**

Identical to Python dataset creation, each SQL line generates 3 entries:
- `context_before`: Current line + preceding SQL clauses
- `context_after`: Current line + following SQL clauses
- `context_full`: Current line + entire SQL query

---

## Models and Architecture

### Pre-trained Models Used

The framework supports multiple pre-trained code models:

1. **microsoft/codebert-base** (Default for detection)
   - Architecture: RoBERTa-based
   - Trained on bimodal data (code + natural language)

2. **nomic-ai/nomic-embed-code** (Default for similarity/embeddings)
   - 7B parameter model
   - Optimized for code embedding tasks
   - Used in line similarity detection and fine-tuning

3. **sentence-transformers/st-codesearch-distilroberta-base**
   - Lightweight alternative (82M parameters)
   - Faster inference, suitable for resource-constrained environments

### Fine-tuned Models

The project fine-tunes embedding models for bug detection:

```
models/
├── finetuned-nomic-embed-code-cosine/   # CosineSimilarityLoss
├── finetuned-nomic-embed-code-mnr/      # MultipleNegativesRankingLoss (RECOMMENDED)
├── finetuned-nomic-embed-code-supcon/   # SupervisedContrastiveLoss
└── finetuned-distilroberta/             # DistilRoBERTa variant
```

### Model Configuration

- **Precision**: FP16 on GPU, FP32 on CPU
- **Max Sequence Length**: 512 tokens
- **LoRA (Low-Rank Adaptation)**:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.1
  - Target modules: q_proj, k_proj, v_proj
  - Trainable parameters: ~0.1-2% of total

---

## Loss Functions

### Why Contrastive Learning?

Contrastive learning is particularly effective for bug detection because:

1. **Semantic Similarity Learning**: The model learns to embed correct code lines closer to their context while pushing buggy lines away, creating a semantic space where similarity indicates correctness.

2. **Handling Class Imbalance**: Buggy lines are typically rare compared to correct lines. Contrastive losses handle this naturally by focusing on relationships rather than raw classification.

3. **Robust Feature Learning**: By contrasting positive and negative pairs, the model learns more discriminative features that generalize better to unseen bugs.

4. **No Hard Negative Mining**: Some contrastive losses (MNR) automatically use in-batch negatives, avoiding the need for explicit negative sampling.

### Implemented Loss Functions

#### 1. CosineSimilarityLoss (Baseline)

```python
train_loss = losses.CosineSimilarityLoss(model)
```

- **Input**: Pairs of (current_line, context) with binary labels
- **Mechanism**: Learns to maximize cosine similarity for correct lines, minimize for buggy lines
- **Use case**: Baseline approach using all data (both buggy and correct)

#### 2. MultipleNegativesRankingLoss (MNR) - RECOMMENDED

```python
train_loss = losses.MultipleNegativesRankingLoss(model)
```

- **Input**: Only positive pairs (correct lines with their contexts)
- **Mechanism**: Uses in-batch negatives automatically
  - Each line-context pair is treated as positive
  - All other contexts in the batch serve as negatives
- **Advantages**:
  - State-of-the-art for embedding learning
  - No need for triplet mining or explicit negative sampling
  - Efficient use of batch for contrastive learning
- **Recommended** for best performance

#### 3. SupervisedContrastiveLoss (SupCon)

```python
from pytorch_metric_learning import losses
supcon_loss = losses.SupConLoss(temperature=0.07)
```

- **Input**: Embeddings with class labels
- **Mechanism**: From SCL-CVD 2024 paper
  - **Inter-class optimization**: Push apart embeddings of different classes (buggy vs. correct)
  - **Intra-class optimization**: Pull together embeddings of same class
- **Temperature**: Controls the sharpness of similarity distribution (default: 0.07)
- **Custom training loop** required with:
  ```python
  combined_embeddings = (line_embeddings + context_embeddings) / 2
  combined_embeddings = F.normalize(combined_embeddings, p=2, dim=1)
  loss = supcon_loss(combined_embeddings, labels)
  ```

#### 4. WeightedEnsembleLoss

```python
ensemble_loss = WeightedEnsembleLoss(
    model=model,
    loss_types=['cosine', 'mnr', 'supcon'],
    weights=[0.3, 0.4, 0.3],
    temperature=0.07
)
```

- **Mechanism**: Combines multiple loss functions with configurable weights
- **Benefits**:
  - Leverages complementary strengths of different losses
  - Provides per-loss logging for analysis
  - Normalizes weights to sum to 1.0
- **Training considerations**:
  - MNR component only uses positive pairs from batch
  - SupCon component requires embedding combination
  - Individual losses are weighted and summed

### Class Weight Handling

To address class imbalance:

```python
def compute_class_weights(dataset):
    class_weights = [
        total_samples / (num_classes * count_buggy),    # Higher weight for buggy
        total_samples / (num_classes * count_correct)   # Lower weight for correct
    ]
    sample_weights = [class_weights[score] for score in scores]
    return class_weights, sample_weights
```

Used with `WeightedRandomSampler` for balanced training.

---

## Evaluation Methodology

### Python Bug Detection Evaluation

The evaluation uses the fine-tuned embedding model to predict whether a line is buggy based on its similarity to context:

```python
def compute_embeddings(model, test_dataset):
    line_embeddings = model.encode(current_lines)
    context_embeddings = model.encode(contexts)
    similarities = cosine_similarity(line_embeddings, context_embeddings)
    # Lower similarity = more likely buggy
```

#### Metrics Computed

1. **Classification Metrics** (at threshold):
   - Accuracy
   - Precision (for buggy class)
   - Recall (for buggy class)
   - F1-Score
   - Confusion Matrix

2. **Ranking Metrics**:
   - ROC-AUC: Area under ROC curve
   - PR-AUC: Area under Precision-Recall curve
   - Spearman Correlation: Between similarity scores and true labels
   - Pearson Correlation: Linear correlation measure

3. **Threshold Optimization**:
   ```python
   def find_optimal_threshold(similarities, true_labels, metric='f1'):
       for thresh in np.arange(0.0, 1.01, 0.01):
           predictions = (similarities >= thresh).astype(int)
           score = f1_score(true_labels, predictions)
           # Track best threshold
   ```

4. **Context Type Analysis**:
   - Separate evaluation for before/after/full contexts
   - Identifies which context type performs best

### SQL Bug Detection Evaluation

#### Benchmark-Based Evaluation

Since NL2SQL-Bugs provides explicit labels for which queries are incorrect:

```python
def evaluate_on_nl2sql_bugs(model, benchmark_data, threshold):
    for entry in benchmark_data:
        query = entry['SQL']
        is_correct = entry['is_correct']  # Ground truth

        # Compute similarity scores for each line
        line_scores = compute_line_similarity_scores(model, query)

        # Predict: has buggy lines?
        has_buggy_lines = any(score < threshold for _, _, score in line_scores)

        # Compare prediction to ground truth
        has_bugs = not is_correct
```

**Advantage**: Since the benchmark explicitly marks which queries are incorrect and often identifies the specific error type, evaluation is straightforward:
- **True Positive**: Model correctly identifies buggy query as having bugs
- **True Negative**: Model correctly identifies correct query as bug-free
- **Type-Specific Accuracy**: Performance breakdown by error category (logical, syntactic, etc.)

#### Metrics

- Overall Accuracy
- Precision, Recall, F1 for bug detection
- Type-specific accuracy for each error category
- Confusion matrix visualization

---

## Applications and Future Directions

### LLM-Guided Bug Fixing

The bug detection system can significantly enhance LLM-based code repair:

1. **Focused Context**: Instead of providing an entire codebase to an LLM for fixing, the system identifies the most likely buggy lines, allowing the LLM to focus on specific problem areas.

2. **SQL Query Debugging**: For SQL queries, the system can help understand:
   - Is the error due to **incorrect filtering** (wrong WHERE clause)?
   - Are there **non-existent column names** being referenced?
   - Are **aliases incorrectly defined or used**?
   - Is the **JOIN logic incorrect**?

3. **Guided Repair Prompts**: LLMs can be prompted with specific information like:
   ```
   "The bug is likely in line 5 (WHERE age >= 18), which appears to have
   an incorrect comparison operator. The correct version likely uses '>'
   instead of '>=' based on the context."
   ```

### Genetic Improvement

The framework is particularly valuable for **Genetic Improvement (GI)** algorithms:

1. **Targeted Mutation**: Rather than randomly selecting code locations for mutation, GI algorithms can prioritize lines flagged as potentially buggy, increasing the probability of generating beneficial mutations.

2. **Efficient Crossover**: Crossover operations can focus on recombining code blocks that contain suspected bugs, rather than random code segments.

3. **Reduced Search Space**: By identifying likely bug locations, the evolutionary search space is dramatically reduced, potentially **decreasing the time to find solutions** by orders of magnitude.

4. **Fitness Guidance**: Uncertainty scores can be incorporated into fitness functions, rewarding mutations that reduce uncertainty (i.e., make code appear more "correct" to the model).

5. **Multi-objective Optimization**: Bug detection scores can serve as an additional objective alongside traditional fitness measures (test passing, performance, etc.).

### Code Review Automation

1. **Pre-commit Hooks**: Automatically flag suspicious code before commits
2. **Pull Request Analysis**: Highlight potentially problematic changes for reviewers
3. **Technical Debt Identification**: Identify code sections that consistently show high uncertainty

### Educational Tools

1. **Learning Platform Integration**: Help students identify errors in their code
2. **Code Quality Metrics**: Provide quantitative measures of code "correctness"
3. **Best Practice Enforcement**: Flag deviations from established coding patterns

---

## Technical Implementation Details

### Core Detection Methods

The `UncertaintyBasedBugDetector` class implements five distinct detection methods:

1. **Semantic Energy**: Uses negative log-likelihood as energy measure
2. **Conformal Prediction**: Statistical guarantees via prediction set sizes
3. **Attention Anomaly Detection**: Analyzes attention pattern irregularities
4. **Token Masking Detection**: Checks if masked tokens are predictable
5. **Line Similarity Detection**: Compares line embeddings to context

### Ensemble Voting

Results from multiple methods are aggregated via voting:

```python
def voting_detection(results, token_vote_threshold=2, line_vote_threshold=2):
    # Lines flagged by >= threshold methods are considered buggy
    # Scores are normalized and averaged across methods
```

### Production Deployment

- **SLURM Scripts**: For HPC cluster deployment
- **Multi-GPU Support**: Distributed training with Accelerate
- **Memory Optimization**: Gradient checkpointing, lazy data loading
- **Checkpoint Management**: Automatic saving of best models

---

## Conclusion

This Uncertainty-Based Bug Detection Framework represents a significant advancement in automated code analysis. By combining multiple uncertainty quantification methods with modern deep learning techniques, it provides a robust foundation for:

1. **Understanding bug locations** in both Python and SQL code
2. **Guiding automated repair systems** with precise bug localization
3. **Accelerating genetic improvement** through targeted search
4. **Enhancing code review processes** with quantitative uncertainty measures

The framework's strength lies in its multi-method ensemble approach, contrastive learning foundation, and practical applicability to real-world software engineering challenges.
