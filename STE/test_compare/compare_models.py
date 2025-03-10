"""
Main script for comparing model performance.
"""

import os
import sys
import fire

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

# Import project modules
from STE.test_compare.config.constants import (
    BASE_MODEL, 
    FINETUNED_MODEL_PATH, 
    TEST_DATASET_PATH, 
    MAX_TURNS,
    DEFAULT_OUTPUT_DIR
)
from STE.test_compare.utility.dataset import load_dataset, load_api_metadata
from STE.test_compare.utility.metrics import calculate_comparison_metrics
from STE.test_compare.evaluation.model_runner import evaluate_test_set
from STE.test_compare.reporting.formatters import save_comparison_results, print_comparison_summary

def main(
    base_model: str = BASE_MODEL,
    finetuned_model_path: str = FINETUNED_MODEL_PATH,
    test_dataset_path: str = TEST_DATASET_PATH,
    max_turns: int = MAX_TURNS,
    output_dir: str = DEFAULT_OUTPUT_DIR
):
    """
    Compare the performance of the base model and the fine-tuned model.
    """
    # Load test dataset
    test_dataset = load_dataset(test_dataset_path)
    if not test_dataset:
        print("No data to evaluate. Exiting.")
        return
        
    # Load API metadata
    api_descriptions, api_list = load_api_metadata()
    
    # Evaluate fine-tuned model
    finetuned_model_results = evaluate_test_set(finetuned_model_path, test_dataset, api_descriptions, api_list, max_turns)
    #finetuned_model_results = evaluate_test_set(finetuned_model_path, test_dataset, max_turns)
    
    # Evaluate base model
    base_model_results = evaluate_test_set(base_model, test_dataset, api_descriptions, api_list, max_turns)
    #base_model_results = evaluate_test_set(base_model, test_dataset, max_turns)
    
    # Prepare comparison results
    comparison_metrics = calculate_comparison_metrics(base_model_results, finetuned_model_results)
    comparison_results = {
        "base_model": base_model_results,
        "finetuned_model": finetuned_model_results,
        "comparison": comparison_metrics
    }
    
    # Save results and print summary
    output_path = save_comparison_results(comparison_results, output_dir)
    print_comparison_summary(comparison_results, output_path)

if __name__ == "__main__":
    fire.Fire(main)