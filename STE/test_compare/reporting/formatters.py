"""
Functions for formatting and saving evaluation results.
"""

import json
import os
import time
from typing import Dict, Any

def save_comparison_results(comparison_results: Dict[str, Any], output_dir: str) -> str:
    """
    Save comparison results to a JSON file
    
    Args:
        comparison_results: Dictionary of comparison results
        output_dir: Directory to save the output
        
    Returns:
        Path to the saved file
    """
    # Create timestamp for the output file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_path = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    return output_path

def print_comparison_summary(comparison_results: Dict[str, Any], output_path: str) -> None:
    """
    Print a summary of the comparison results
    
    Args:
        comparison_results: Dictionary of comparison results
        output_path: Path where results were saved
    """
    base_results = comparison_results["base_model"]
    finetuned_results = comparison_results["finetuned_model"]
    
    print("\n=== Comparison Results ===")
    print(f"Base model: {base_results['model']}")
    print(f"  Successful queries: {base_results['metrics']['successful_queries']}/{base_results['metrics']['total_queries']} "
          f"({base_results['metrics']['success_rate']*100:.2f}%)")
    print(f"  Average turns for successful queries: {base_results['metrics']['avg_turns_successful']:.2f}")
    
    print(f"\nFine-tuned model: {finetuned_results['model']}")
    print(f"  Successful queries: {finetuned_results['metrics']['successful_queries']}/{finetuned_results['metrics']['total_queries']} "
          f"({finetuned_results['metrics']['success_rate']*100:.2f}%)")
    print(f"  Average turns for successful queries: {finetuned_results['metrics']['avg_turns_successful']:.2f}")
    
    print(f"\nSuccess rate improvement: {comparison_results['comparison']['success_rate_diff']*100:.2f}%")
    
    turn_diff = comparison_results['comparison']['avg_turns_diff']
    better_worse = "fewer" if turn_diff < 0 else "more"
    print(f"Average turns difference: {abs(turn_diff):.2f} {better_worse} turns")
    
    print(f"\nResults saved to {output_path}")