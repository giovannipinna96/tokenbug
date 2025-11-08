"""
Example usage and testing script for the Uncertainty-Based Bug Detector
Demonstrates detection on various bug patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bug_detection import UncertaintyBasedBugDetector, analyze_code
import pandas as pd
from typing import Dict, List


# Collection of buggy code examples for testing
BUG_EXAMPLES = {
    "off_by_one": {
        "buggy": """def binary_search(arr, target):
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
    return -1""",
        "fixed": """def binary_search(arr, target):
    left = 0
    right = len(arr) - 1  # Fixed
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
        "bug_line": 2
    },
    
    "missing_base_case": {
        "buggy": """def factorial(n):
    # Bug: missing base case
    return n * factorial(n - 1)""",
        "fixed": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
        "bug_line": 1
    },
    
    "incorrect_operator": {
        "buggy": """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total *= num  # Bug: should be +=
    return total / len(numbers)""",
        "fixed": """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num  # Fixed
    return total / len(numbers)""",
        "bug_line": 3
    },
    
    "type_mismatch": {
        "buggy": """def process_data(data):
    result = []
    for item in data:
        result.append(item + 1)
    return ''.join(result)  # Bug: trying to join integers""",
        "fixed": """def process_data(data):
    result = []
    for item in data:
        result.append(str(item + 1))
    return ''.join(result)  # Fixed: converting to strings""",
        "bug_line": 4
    },
    
    "loop_condition": {
        "buggy": """def count_occurrences(lst, target):
    count = 0
    i = 0
    while i <= len(lst):  # Bug: should be i < len(lst)
        if lst[i] == target:
            count += 1
        i += 1
    return count""",
        "fixed": """def count_occurrences(lst, target):
    count = 0
    i = 0
    while i < len(lst):  # Fixed
        if lst[i] == target:
            count += 1
        i += 1
    return count""",
        "bug_line": 3
    }
}


def visualize_detection_results(results: Dict, code: str, title: str = "Bug Detection Results"):
    """
    Create visualizations for detection results
    
    Args:
        results: Results from analyze_code
        code: Original code
        title: Title for the visualization
    """
    lines = code.strip().split('\n')
    num_lines = len(lines)
    methods = list(results['detailed_results'].keys())
    
    # Create heatmap data
    heatmap_data = np.zeros((len(methods), num_lines))
    
    for i, method in enumerate(methods):
        if results['detailed_results'][method]:
            result = results['detailed_results'][method]
            for line_num, _, score in result.anomaly_lines:
                if line_num < num_lines:
                    heatmap_data[i, line_num] = score
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=[f"L{i+1}" for i in range(num_lines)],
                yticklabels=methods,
                cmap='YlOrRd',
                cbar_kws={'label': 'Anomaly Score'},
                ax=axes[0])
    axes[0].set_title(f"{title} - Anomaly Scores by Method")
    axes[0].set_xlabel("Line Number")
    axes[0].set_ylabel("Detection Method")
    
    # Bar plot of consensus scores
    if results['consensus_lines']:
        line_nums = [info['line_number'] for info in results['consensus_lines']]
        scores = [info['avg_score'] for info in results['consensus_lines']]
        num_methods = [info['num_methods'] for info in results['consensus_lines']]
        
        bars = axes[1].bar(range(len(line_nums)), scores, color='coral')
        axes[1].set_xticks(range(len(line_nums)))
        axes[1].set_xticklabels([f"Line {n+1}" for n in line_nums])
        axes[1].set_xlabel("Detected Anomalous Lines")
        axes[1].set_ylabel("Average Anomaly Score")
        axes[1].set_title("Consensus Anomalies (Detected by Multiple Methods)")
        
        # Add number of methods as text on bars
        for bar, nm in zip(bars, num_methods):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{nm} methods',
                        ha='center', va='bottom', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No consensus anomalies detected", 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Consensus Anomalies")
    
    plt.tight_layout()
    return fig


def compare_buggy_vs_fixed(detector: UncertaintyBasedBugDetector, 
                          buggy_code: str, 
                          fixed_code: str,
                          bug_name: str) -> Dict:
    """
    Compare detection results between buggy and fixed versions
    
    Args:
        detector: Bug detector instance
        buggy_code: Buggy version of code
        fixed_code: Fixed version of code
        bug_name: Name of the bug pattern
        
    Returns:
        Comparison statistics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {bug_name}")
    print(f"{'='*60}")
    
    # Analyze both versions
    buggy_results = detector.ensemble_detection(buggy_code)
    fixed_results = detector.ensemble_detection(fixed_code)
    
    comparison = {
        'bug_name': bug_name,
        'methods': {},
        'improvement': {}
    }
    
    for method in buggy_results.keys():
        if buggy_results[method] and fixed_results[method]:
            buggy_score = np.mean(buggy_results[method].anomaly_scores)
            fixed_score = np.mean(fixed_results[method].anomaly_scores)
            
            buggy_anomalies = len(buggy_results[method].anomaly_tokens)
            fixed_anomalies = len(fixed_results[method].anomaly_tokens)
            
            comparison['methods'][method] = {
                'buggy_mean_score': buggy_score,
                'fixed_mean_score': fixed_score,
                'buggy_anomalies': buggy_anomalies,
                'fixed_anomalies': fixed_anomalies,
                'score_reduction': buggy_score - fixed_score,
                'anomaly_reduction': buggy_anomalies - fixed_anomalies
            }
            
            print(f"\n{method}:")
            print(f"  Buggy - Mean Score: {buggy_score:.4f}, Anomalies: {buggy_anomalies}")
            print(f"  Fixed - Mean Score: {fixed_score:.4f}, Anomalies: {fixed_anomalies}")
            print(f"  Improvement: Score↓ {buggy_score - fixed_score:.4f}, "
                  f"Anomalies↓ {buggy_anomalies - fixed_anomalies}")
    
    # Calculate overall improvement
    avg_score_reduction = np.mean([m['score_reduction'] for m in comparison['methods'].values()])
    avg_anomaly_reduction = np.mean([m['anomaly_reduction'] for m in comparison['methods'].values()])
    
    comparison['improvement'] = {
        'avg_score_reduction': avg_score_reduction,
        'avg_anomaly_reduction': avg_anomaly_reduction
    }
    
    print(f"\nOverall Improvement:")
    print(f"  Average Score Reduction: {avg_score_reduction:.4f}")
    print(f"  Average Anomaly Reduction: {avg_anomaly_reduction:.2f}")
    
    return comparison


def run_comprehensive_test(model_name: str = "microsoft/codebert-base"):
    """
    Run comprehensive test on all bug examples
    
    Args:
        model_name: Model to use for detection
    """
    print(f"Initializing detector with model: {model_name}")
    detector = UncertaintyBasedBugDetector(model_name)
    
    all_comparisons = []
    
    for bug_name, example in BUG_EXAMPLES.items():
        comparison = compare_buggy_vs_fixed(
            detector, 
            example['buggy'], 
            example['fixed'],
            bug_name
        )
        all_comparisons.append(comparison)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL BUG PATTERNS")
    print(f"{'='*60}")
    
    # Create summary dataframe
    summary_data = []
    for comp in all_comparisons:
        for method, stats in comp['methods'].items():
            summary_data.append({
                'Bug Pattern': comp['bug_name'],
                'Method': method,
                'Score Reduction': stats['score_reduction'],
                'Anomaly Reduction': stats['anomaly_reduction']
            })
    
    df = pd.DataFrame(summary_data)
    
    # Print method performance
    print("\nMethod Performance (Average across all bugs):")
    method_summary = df.groupby('Method').agg({
        'Score Reduction': 'mean',
        'Anomaly Reduction': 'mean'
    }).round(4)
    print(method_summary)
    
    # Print bug pattern summary
    print("\nBug Pattern Detection (Average across all methods):")
    bug_summary = df.groupby('Bug Pattern').agg({
        'Score Reduction': 'mean',
        'Anomaly Reduction': 'mean'
    }).round(4)
    print(bug_summary)
    
    # Find best method for each bug type
    print("\nBest Method for Each Bug Pattern:")
    for bug in BUG_EXAMPLES.keys():
        bug_data = df[df['Bug Pattern'] == bug]
        best_method = bug_data.loc[bug_data['Score Reduction'].idxmax(), 'Method']
        best_score = bug_data['Score Reduction'].max()
        print(f"  {bug}: {best_method} (score reduction: {best_score:.4f})")
    
    return df


def test_single_code_sample():
    """Test on a single code sample with visualization"""
    
    code = """def find_max(numbers):
    if not numbers:
        return None
    
    max_val = numbers[0]
    for i in range(len(numbers)):  # Could use enumerate
        if numbers[i] > max_val:
            max_value = numbers[i]  # Bug: typo in variable name
    return max_val"""
    
    print("Analyzing code for bugs...")
    results = analyze_code(code)
    
    print("\n=== Detection Results ===")
    print(f"Methods detecting anomalies: {results['methods_detected_anomalies']}/{results['total_methods']}")
    
    if results['consensus_tokens']:
        print("\nConsensus Anomalous Tokens:")
        for token in results['consensus_tokens'][:5]:  # Top 5
            print(f"  Pos {token['position']}: '{token['token']}' "
                  f"(score: {token['avg_score']:.3f}, methods: {token['num_methods']})")
    
    if results['consensus_lines']:
        print("\nConsensus Anomalous Lines:")
        for line in results['consensus_lines']:
            print(f"  Line {line['line_number'] + 1}: {line['line'].strip()}")
            print(f"    Score: {line['avg_score']:.3f}, Methods: {line['num_methods']}")
    
    # Visualize results
    fig = visualize_detection_results(results, code, "Variable Name Bug Detection")
    plt.savefig('bug_detection_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to bug_detection_visualization.png")
    
    return results


if __name__ == "__main__":
    print("Bug Detection Framework - Comprehensive Testing")
    print("=" * 60)
    
    # Test single code sample
    print("\n1. Testing Single Code Sample")
    test_single_code_sample()
    
    # Run comprehensive test on all bug patterns
    print("\n2. Running Comprehensive Test on Multiple Bug Patterns")
    df_results = run_comprehensive_test()
    
    # Save results
    df_results.to_csv('bug_detection_results.csv', index=False)
    print("\nResults saved to bug_detection_results.csv")
    
    print("\n✓ Testing complete!")
