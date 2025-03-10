"""
Functions for evaluating models on queries.
"""

import os
import json
from typing import Dict, List, Any
from tqdm import tqdm
from datetime import datetime

from STE.my_llm import chat_my
from STE.test_compare.evaluation.action_parser import parse_response
from STE.test_compare.evaluation.evaluation_funct import run_evaluation
from STE.test_compare.utility.dataset import load_api_metadata

from config.constants import TEMPERATURE, MAX_TURNS

def evaluate_query_with_model(model_name: str, 
                             query_item: Dict[str, str],
                             api_descriptions: Dict[str, str], 
                             api_list: List[str],
                             max_turns: int = MAX_TURNS) -> Dict[str, Any]:
    """
    Evaluate a single query with the specified model.
    
    Args:
        model_name: Name of the model to use
        query_item: Dictionary with 'query' and 'expected_answer' keys
        api_descriptions: Dictionary of API descriptions
        api_list: List of available APIs
        max_turns: Maximum number of turns allowed
    
    Returns:
        Dictionary with evaluation results
    """
    query = query_item.get("query", "")
    expected_answer = query_item.get("expected_answer", "")
    
    # Initialize messages with the query
    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in evaluating text using metrics."},
        {"role": "user", "content": query}
    ]
    
    query_result = {
        "query": query,
        "expected_answer": expected_answer,
        "turns": [],
        "success": False,
        "solved_at_turn": -1
    }
    
    # First response from model
    os.environ["MODEL_NAME"] = model_name  # Set model for the chat_my function
    response_messages, is_response_empty = chat_my(messages, "", temp=TEMPERATURE, visualize=False, max_tokens=2048)
    model_response = response_messages[-1]["content"] if len(response_messages) > 1 else ""
    
    # Process each turn
    for turn in range(max_turns + 1):
        print(f"DEBUG: Processing turn {turn} for {model_name}")
        
        # Parse the model response to extract actions
        parsed_response = parse_response(
            model_response, 
            api_list, 
            api_descriptions,
            proc_thought=False,
            check_API_name=True
        )
        
        # Record this turn's information
        turn_info = {
            "turn": turn,
            "model_response": model_response,
            "parsed_response": parsed_response,
            "evaluation_results": []
        }
        
        # Check for final answer - this needs to happen first since it could be in the initial response
        # if parsed_response.get('finish', False) or "Final Answer:" in model_response:
        #     print(f"DEBUG: Final answer detected at turn {turn}")
        #     query_result["success"] = True
        #     query_result["solved_at_turn"] = turn
        #     turn_info["evaluation_results"].append("Final answer reached")
        #     query_result["turns"].append(turn_info)
        #     break
            
        # Check if parsing was successful
        if not parsed_response.get('parse_successful', False):
            error_msg = parsed_response.get('parse_error_msg', 'Failed to parse actions and inputs from the response')
            print(f"DEBUG: {error_msg}")
            turn_info["evaluation_results"] = [error_msg]
            query_result["turns"].append(turn_info)
            
            # Send the error back to the model for next turn
            evaluation_result = f"Parsing Error: {error_msg}"
            messages.append({"role": "user", "content": evaluation_result})
            # Try to get next response if we haven't reached max turns
            if turn < max_turns - 1:
                try:
                    response_messages, _ = chat_my(messages, "", temp=TEMPERATURE, visualize=False, max_tokens=2048)
                    model_response = response_messages[-1]["content"] if len(response_messages) > 1 else ""
                    print("MODEL RESPONSE: ", model_response)
                except Exception as e:
                    print(f"DEBUG: Error getting model response after parse error: {str(e)}")
                    break
            continue
        
        # Run evaluations for each action
        evaluation_result = ""
        evaluation_success = True  # Assume success until proven otherwise
        all_evaluations = []
        
        for act in parsed_response.get('actions', []):
            try:
                # Adjust based on what run_evaluation actually returns
                # If it returns a tuple (result, evaluated), use both values
                # If it returns just a result string, use that
                try:
                    # Try unpacking as tuple first
                    result = run_evaluation(act['action'], act['action_input'], api_list, api_descriptions)
                    
                    # Check if result indicates an error
                    has_error = isinstance(result, str) and ("error" in result.lower() or 
                                                          "incorrect" in result.lower() or 
                                                          "invalid" in result.lower())
                    
                    if has_error:
                        evaluation_success = False
                        
                    turn_info["evaluation_results"].append(result)
                    all_evaluations.append(result)
                    evaluation_result += f"Evaluation Result: {result}\n"
                    
                except ValueError:
                    # If unpacking fails, assume it returns just one value
                    result = run_evaluation(act['action'], act['action_input'], api_list, api_descriptions)
                    turn_info["evaluation_results"].append(result)
                    all_evaluations.append(result)
                    evaluation_result += f"Evaluation Result: {result}\n"
                    
            except Exception as e:
                error_msg = f"Error during evaluation: {str(e)}"
                turn_info["evaluation_results"].append(error_msg)
                evaluation_result += f"Evaluation Result: {error_msg}\n"
                evaluation_success = False
        
        # I go to the next query if evaluation_success is true, since the model has finished the evaluation
        if evaluation_success:
            print(f"DEBUG: All evaluations successful at turn {turn}")
            evaluation_result += "\nAll evaluations completed successfully. Please provide your final answer using:\nThought: I now know the final answer\nFinal Answer: [your answer]"
            query_result["success"] = True
            query_result["solved_at_turn"] = turn
            turn_info["evaluation_results"].append("Final answer reached")
            query_result["turns"].append(turn_info)
            break
        # Save the turn information with all evaluation results
        query_result["turns"].append(turn_info)
        
        # # If all evaluations were successful, prompt model to provide final answer
        # if evaluation_success and all_evaluations:
        #     print(f"DEBUG: All evaluations successful at turn {turn}, prompting for final answer")
        #     evaluation_result += "\nAll evaluations completed successfully. Please provide your final answer using:\nThought: I now know the final answer\nFinal Answer: [your answer]"
        # else:
        evaluation_result += "\nPlease fix any errors and try again."
            
        messages.append({"role": "user", "content": evaluation_result.strip()})
        # Get the next response from the model if we haven't finished
        if turn < max_turns - 1:
            try:
                response_messages, _ = chat_my(messages, "", temp=TEMPERATURE, visualize=False, max_tokens=2048)
                model_response = response_messages[-1]["content"] if len(response_messages) > 1 else ""
                
                # Check if the new response has a final answer
                if "Final Answer:" in model_response:
                    print(f"DEBUG: Model provided final answer after successful evaluations")
                    # We'll catch this in the next iteration at the final answer check
                    
            except Exception as e:
                print(f"DEBUG: Error getting model response: {str(e)}")
                turn_info = {
                    "turn": turn,
                    "error": str(e),
                    "evaluation_results": [f"Error: {str(e)}"]
                }
                query_result["turns"].append(turn_info)
                break
    
    # Check if we've exhausted max turns without success
    if len(query_result["turns"]) >= max_turns and not query_result["success"]:
        print(f"DEBUG: Reached maximum turns ({max_turns}) without a successful completion")
    
    return query_result

def calculate_metrics(queries):
    """Calculate aggregate metrics from query results."""
    total_queries = len(queries)
    successful_queries = sum(1 for q in queries if q["success"])
    avg_turns_successful = 0
    if successful_queries > 0:
        avg_turns_successful = sum(q["solved_at_turn"] + 1 for q in queries if q["success"]) / successful_queries
    
    return {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
        "avg_turns_successful": avg_turns_successful
    }

def evaluate_test_set(model_name: str, 
                     test_dataset: List[Dict[str, str]], 
                     api_descriptions: Dict[str, str], 
                     api_list: List[str],
                     max_turns: int = MAX_TURNS) -> Dict[str, Any]:
    """
    Evaluate a model on the test dataset.
    
    Args:
        model_name: Name or path of the model to evaluate
        test_dataset: Test dataset list
        api_descriptions: Dictionary of API descriptions
        api_list: List of available APIs
        max_turns: Maximum number of turns for each query
        
    Returns:
        Dictionary containing evaluation results
    """
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "queries": []
    }
    
    print(f"\nEvaluating model: {model_name}")
    
    # Process each query in the test dataset
    for i, query_item in enumerate(tqdm(test_dataset, desc=f"Evaluating {model_name}")):
        print(f"\nProcessing query {i+1}/{len(test_dataset)}: {query_item.get('query', '')[:50]}...")
        
        # Evaluate the query - we pass all parameters including api_descriptions and api_list
        query_result = evaluate_query_with_model(
            model_name, query_item, api_descriptions, api_list, max_turns
        )
        
        # Add to results
        results["queries"].append(query_result)
        
        # Print brief outcome
        success_status = "✓" if query_result["success"] else "✗"
        print("\n\n-------------------------")
        print(f"Result: {success_status} (solved at turn {query_result['solved_at_turn'] + 1 if query_result['solved_at_turn'] >= 0 else 'N/A'})")
        print("\n\n-------------------------")

    # Calculate aggregate metrics
    results["metrics"] = calculate_metrics(results["queries"])
    
    return results