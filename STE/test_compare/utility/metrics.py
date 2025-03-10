"""
Functions for calculating evaluation metrics.
"""

from typing import List, Dict, Any

def calculate_metrics(query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics from query results
    
    Args:
        query_results: List of query result dictionaries
        
    Returns:
        Dictionary of metrics
    """
    total_queries = len(query_results)
    successful_queries = sum(1 for q in query_results if q["success"])
    avg_turns_successful = 0
    
    if successful_queries > 0:
        avg_turns_successful = sum(q["solved_at_turn"] + 1 for q in query_results if q["success"]) / successful_queries
    
    return {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
        "avg_turns_successful": avg_turns_successful
    }

def calculate_comparison_metrics(base_results: Dict[str, Any], 
                                finetuned_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comparison metrics between base and finetuned model results
    
    Args:
        base_results: Base model results dictionary
        finetuned_results: Finetuned model results dictionary
        
    Returns:
        Dictionary of comparison metrics
    """
    return {
        "success_rate_diff": finetuned_results["metrics"]["success_rate"] - base_results["metrics"]["success_rate"],
        "avg_turns_diff": finetuned_results["metrics"]["avg_turns_successful"] - base_results["metrics"]["avg_turns_successful"]
    }
