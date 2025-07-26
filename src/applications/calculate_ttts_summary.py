#!/usr/bin/env python3
"""
Calculate summary statistics for TTTS evaluation results
"""
import json
import numpy as np
from pathlib import Path

def calculate_summary_stats(results_file):
    """Calculate summary statistics from TTTS results"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("TTTS Evaluation Summary")
    print("="*80)
    
    for dataset_name, dataset_results in data.items():
        print(f"\nDataset: {dataset_name}")
        print("-"*40)
        
        # Extract accuracy scores
        accuracies = []
        thinking_tokens = []
        
        for token, result in dataset_results.items():
            acc = result.get('acc', 0)
            accuracies.append(acc)
            thinking_tokens.append(token)
        
        # Sort by accuracy
        sorted_results = sorted(zip(thinking_tokens, accuracies), 
                              key=lambda x: x[1], reverse=True)
        
        # Calculate statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        median_acc = np.median(accuracies)
        
        print(f"\n📊 Summary Statistics:")
        print(f"  - Total thinking tokens evaluated: {len(thinking_tokens)}")
        print(f"  - Mean accuracy: {mean_acc:.1f}%")
        print(f"  - Std deviation: {std_acc:.1f}%")
        print(f"  - Median accuracy: {median_acc:.1f}%")
        print(f"  - Max accuracy: {max_acc:.1f}%")
        print(f"  - Min accuracy: {min_acc:.1f}%")
        
        print(f"\n🏆 Top 5 Thinking Tokens:")
        for i, (token, acc) in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {token}: {acc:.1f}%")
        
        print(f"\n📉 Bottom 5 Thinking Tokens:")
        for i, (token, acc) in enumerate(sorted_results[-5:], 1):
            print(f"  {i}. {token}: {acc:.1f}%")
        
        # Group by accuracy ranges
        print(f"\n📈 Accuracy Distribution:")
        ranges = [(40, 100, "40%+"), (35, 40, "35-40%"), (30, 35, "30-35%"), 
                  (25, 30, "25-30%"), (0, 25, "<25%")]
        
        for low, high, label in ranges:
            count = sum(1 for _, acc in sorted_results if low <= acc < high)
            if count > 0:
                tokens = [token for token, acc in sorted_results if low <= acc < high]
                print(f"  - {label}: {count} tokens ({', '.join(tokens[:5])}{'...' if count > 5 else ''})")
        
        # Calculate overall score (weighted average if multiple samples)
        num_samples = dataset_results[thinking_tokens[0]].get('num_samples', 30)
        total_score = mean_acc
        
        print(f"\n🎯 Overall Dataset Score: {total_score:.1f}%")
        print(f"   (Based on {num_samples} test samples per thinking token)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "/root/llm_eval/MI-Peaks/src/applications/results/DeepSeek-R1-Distill-Llama-8B/aime24_budget4096/TTTS_results.json"
    
    calculate_summary_stats(results_file)