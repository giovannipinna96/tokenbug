# Uncertainty-Based Bug Detection Framework

Implementation of uncertainty quantification methods for detecting buggy code, based on the paper "Advanced Uncertainty-Based Error Detection Methods for Targeted Genetic Improvement of Software".

## Overview

This framework implements multiple uncertainty-based methods to detect potentially buggy tokens and lines in source code:

1. **Semantic Energy**: Uses pre-softmax logits to compute energy scores
2. **Conformal Prediction**: Provides statistical guarantees for uncertainty quantification
3. **Attention Anomaly Detection**: Analyzes attention patterns (entropy, self-attention, variance)
4. **Token Masking Detection**: Masks tokens and checks if original is in top-k predictions
5. **Line Similarity Detection**: Uses cosine similarity to detect anomalous lines

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from bug_detection import UncertaintyBasedBugDetector, analyze_code

# Simple analysis
code = """
def binary_search(arr, target):
    left = 0
    right = len(arr)  # Bug: should be len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

# Analyze with all methods
results = analyze_code(code)

# Print consensus anomalies
print("Anomalous lines detected by multiple methods:")
for line_info in results['consensus_lines']:
    print(f"Line {line_info['line_number']}: {line_info['line'].strip()}")
```

### Using Individual Methods

```python
from bug_detection import UncertaintyBasedBugDetector

# Initialize detector with specific model
detector = UncertaintyBasedBugDetector(model_name="microsoft/codebert-base")

# Use specific methods
semantic_result = detector.semantic_energy(code)
conformal_result = detector.conformal_prediction(code)
attention_result = detector.attention_anomaly_detection(code)
masking_result = detector.token_masking_detection(code)
similarity_result = detector.line_similarity_detection(code)

# Access results
for position, token, score in semantic_result.anomaly_tokens:
    print(f"Anomalous token at position {position}: '{token}' (score: {score:.3f})")
```

### Ensemble Detection

```python
# Run ensemble of methods
detector = UncertaintyBasedBugDetector()
results = detector.ensemble_detection(code, methods=['semantic_energy', 'attention_anomaly'])

# Process results
for method_name, result in results.items():
    if result:
        print(f"\n{method_name} detected {len(result.anomaly_tokens)} anomalous tokens")
```

## Methods Explained

### 1. Semantic Energy

Operates on pre-softmax logits to capture uncertainty before normalization:
- **Formula**: `Energy(xi) = -logit(xi)`
- **Advantage**: Better separability between correct/incorrect predictions
- **Best for**: Semantic errors that don't manifest as syntactic anomalies

### 2. Conformal Prediction

Provides formal statistical guarantees for uncertainty:
- **Coverage guarantee**: `P(ytrue ∈ Cα(x)) ≥ 1 − α`
- **Uncertainty measure**: Size of prediction set
- **Best for**: Critical applications requiring reliability

### 3. Attention Anomaly Detection

Analyzes attention patterns to identify unusual behavior:
- **Metrics**: Entropy, self-attention, variance
- **Combined score**: Weighted combination of normalized metrics
- **Best for**: Structural errors and control flow issues

### 4. Token Masking Detection

Masks tokens and checks if original is in top predictions:
- **Process**: Mask each token, predict, check if original in top-k
- **Anomaly**: Original token not in top predictions
- **Best for**: Typos and unexpected token usage

### 5. Line Similarity Detection

Uses embeddings to find lines dissimilar to context:
- **Comparison**: Previous lines, next lines, rest of function
- **Metric**: Cosine similarity of line embeddings
- **Best for**: Logic errors and inconsistent code patterns

## Supported Models

The framework supports various model architectures:

### Recommended Models
- **microsoft/codebert-base**: General purpose code understanding
- **microsoft/graphcodebert-base**: With data flow understanding
- **Salesforce/codet5p-220m**: Efficient encoder-decoder model
- **deepseek-ai/deepseek-coder-1.3b-base**: Specialized for code
- **bigcode/starcoderbase-1b**: Fast and efficient

### Model Selection

```python
# Small, fast model
detector = UncertaintyBasedBugDetector("microsoft/codebert-base")

# Larger, more accurate model
detector = UncertaintyBasedBugDetector("deepseek-ai/deepseek-coder-6.7b-base")

# Custom device selection
detector = UncertaintyBasedBugDetector("microsoft/codebert-base", device="cuda")
```

## Running Tests

Test the framework on various bug patterns:

```bash
python test_bug_detection.py
```

This will:
1. Test on a single code sample with visualization
2. Run comprehensive tests on multiple bug patterns
3. Compare buggy vs fixed code versions
4. Generate performance statistics

## Output Format

### Detection Result Structure

```python
@dataclass
class DetectionResult:
    method: str                    # Method name
    anomaly_scores: List[float]    # Score for each token/line
    anomaly_tokens: List[Tuple[int, str, float]]  # (position, token, score)
    anomaly_lines: List[Tuple[int, str, float]]   # (line_number, line, score)
    threshold: float               # Detection threshold used
    metadata: Dict[str, Any]       # Method-specific information
```

### Analysis Summary

```python
{
    'total_methods': 5,
    'methods_detected_anomalies': 3,
    'consensus_tokens': [
        {
            'position': 42,
            'token': 'len',
            'avg_score': 0.82,
            'num_methods': 3
        }
    ],
    'consensus_lines': [
        {
            'line_number': 2,
            'line': 'right = len(arr)',
            'avg_score': 0.75,
            'num_methods': 2
        }
    ],
    'detailed_results': {...}
}
```

## Performance Considerations

### Memory Optimization
- Uses FP16 precision on GPU when available
- Implements gradient checkpointing for large models
- Processes code in chunks for long files

### Speed Optimization
- Batch processing for multiple files
- Cached model loading
- Selective method execution

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB VRAM (for small models)
- **Recommended**: 16GB RAM, 8GB VRAM
- **CPU-only mode**: Available but slower

## Limitations

1. **Context Length**: Limited by model's max sequence length (typically 512 tokens)
2. **Language Support**: Best performance on Python and Java
3. **Complex Bugs**: May miss logic errors requiring deep semantic understanding
4. **False Positives**: Unusual but correct code patterns may be flagged

## Advanced Usage

### Custom Thresholds

```python
# Adjust sensitivity
result = detector.semantic_energy(code, k=2.0)  # Higher k = fewer detections

# Custom confidence level for conformal prediction
result = detector.conformal_prediction(code, alpha=0.05)  # 95% confidence
```

### Filtering Results

```python
# Filter high-confidence anomalies only
high_confidence = [
    (pos, token, score) 
    for pos, token, score in result.anomaly_tokens 
    if score > result.threshold * 1.5
]
```

### Combining with Static Analysis

```python
# Use as pre-filter for expensive static analysis
if len(results['consensus_lines']) > 0:
    # Run static analysis only on suspected lines
    for line_info in results['consensus_lines']:
        run_static_analysis(line_info['line_number'])
```

## Citation

Based on the paper:
```
Advanced Uncertainty-Based Error Detection Methods for 
Targeted Genetic Improvement of Software
Anonymous Author(s), 2025
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional programming languages
- Integration with IDEs
- Real-time detection during coding
- Improved visualization tools
