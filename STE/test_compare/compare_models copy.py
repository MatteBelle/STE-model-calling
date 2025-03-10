import json
import os
import time
from datetime import datetime
import sys
import fire
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Add STE directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

# Import STE functions
from STE.my_llm import chat_my
from STE.utils import parse_response
from STE.main import run_evaluation, run_llm_judge_evaluation

# Define constants
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FINETUNED_MODEL_PATH = "finetuning/outputs/final_model"
TEST_DATASET_PATH = "finetuning/test_dataset.json"
TEMPERATURE = 0.6
MAX_TURNS = 3

def load_dataset(filepath: str) -> List[Dict[str, str]]:
    """
    Load the dataset from a JSON file.
    
    Args:
        filepath: Path to the dataset JSON file
        
    Returns:
        List of query dictionaries
    """
    print(f"Loading dataset from {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return []

def load_api_metadata():
    """
    Load API descriptions and API list.
    
    Returns:
        Tuple of (API_descriptions, API_list)
    """
    with open("STE/tool_metadata/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)
    
    with open("STE/tool_metadata/API_list.json", "r", encoding='utf-8') as f:
        API_list = json.load(f)
        
    return API_descriptions, API_list

def extract_action_and_input(response: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Extract action and action input from model response.
    
    Args:
        response: Model's response text
        
    Returns:
        Tuple of (action, action_input)
    """
    try:
        action_start = response.find("Action:")
        if action_start == -1:
            return None, None
        
        action_input_start = response.find("Action Input:", action_start)
        if action_input_start == -1:
            return None, None
        
        action = response[action_start + len("Action:"):action_input_start].strip()
        action_input_text = response[action_input_start + len("Action Input:"):].strip()
        
        # Parse the JSON input
        try:
            action_input = json.loads(action_input_text)
            return action, action_input
        except json.JSONDecodeError:
            return action, {"error": "Invalid JSON in action input", "raw_text": action_input_text}
    except Exception as e:
        print(f"Error extracting action and input: {e}")
        return None, None

def evaluate_query_with_model(model_name: str, query_item: Dict[str, str], 
                              api_descriptions: Dict[str, str], api_list: List[str],
                              max_turns: int) -> Dict[str, Any]:
    """
    Evaluate a single query using the specified model.
    
    Args:
        model_name: Name or path of the model to evaluate
        query_item: Query dictionary
        api_descriptions: Dictionary of API descriptions
        api_list: List of available APIs
        max_turns: Maximum number of turns
        
    Returns:
        Dictionary containing evaluation results
    """
    # Extract system message and query
    system_msg = query_item.get("system", "You are a bot that creates and responds to evaluation queries.")
    query = query_item.get("query", "")
    expected_answer = query_item.get("answer", "")
    
    # Initialize messages
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]
    
    # Initialize response tracking
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
    for turn in range(max_turns):
        # Extract action and input from the model's response
        action, action_input = extract_action_and_input(model_response)
        
        # Record this turn's information
        turn_info = {
            "turn": turn,
            "model_response": model_response,
            "action": action,
            "action_input": action_input,
        }
        
        # If we couldn't parse the response correctly
        if action is None or action_input is None:
            turn_info["evaluation_result"] = "Failed to parse action and input from the response."
            query_result["turns"].append(turn_info)
            break
        
        # Run the evaluation using the extracted action and input
        try:
            evaluation_result = run_evaluation(action, action_input, api_list, api_descriptions)
            turn_info["evaluation_result"] = evaluation_result
            
            # Add the evaluation result to the messages for the next turn
            messages.append({"role": "user", "content": f"Evaluation Result: {evaluation_result}"})
            
            # Get the next response from the model
            response_messages, is_response_empty = chat_my(messages, "", temp=TEMPERATURE, 
                                                         visualize=False, max_tokens=2048)
            model_response = response_messages[-1]["content"] if len(response_messages) > 1 else ""
            
            # Save the turn information
            query_result["turns"].append(turn_info)
            
            # Check if the model's response now indicates a successful completion
            # (This is a simplified check - in real implementation you would need to define success criteria)
            if "finish" in model_response.lower() or "final answer" in model_response.lower():
                query_result["success"] = True
                query_result["solved_at_turn"] = turn
                break
                
        except Exception as e:
            turn_info["error"] = str(e)
            turn_info["evaluation_result"] = f"Error: {str(e)}"
            query_result["turns"].append(turn_info)
            break
    
    return query_result

def evaluate_test_set(model_name: str, test_dataset: List[Dict[str, str]], 
                     api_descriptions: Dict[str, str], api_list: List[str], 
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
        
        # Evaluate the query
        query_result = evaluate_query_with_model(
            model_name, query_item, api_descriptions, api_list, max_turns
        )
        
        # Add to results
        results["queries"].append(query_result)
        
        # Print brief outcome
        success_status = "✓" if query_result["success"] else "✗"
        print(f"Result: {success_status} (solved at turn {query_result['solved_at_turn'] + 1 if query_result['solved_at_turn'] >= 0 else 'N/A'})")
    
    # Calculate aggregate metrics
    total_queries = len(results["queries"])
    successful_queries = sum(1 for q in results["queries"] if q["success"])
    avg_turns_successful = 0
    if successful_queries > 0:
        avg_turns_successful = sum(q["solved_at_turn"] + 1 for q in results["queries"] if q["success"]) / successful_queries
    
    results["metrics"] = {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
        "avg_turns_successful": avg_turns_successful
    }
    
    return results

def main(
    base_model: str = BASE_MODEL,
    finetuned_model_path: str = FINETUNED_MODEL_PATH,
    test_dataset_path: str = TEST_DATASET_PATH,
    max_turns: int = MAX_TURNS,
    output_dir: str = "finetuning/outputs"
):
    """
    Compare the performance of the base model and the fine-tuned model.
    
    Args:
        base_model: Name of the base model
        finetuned_model_path: Path to the fine-tuned model
        test_dataset_path: Path to the test dataset
        max_turns: Maximum number of turns for each query
        output_dir: Directory to save the output
    """
    # Create timestamp for the output file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test dataset
    test_dataset = load_dataset(test_dataset_path)
    if not test_dataset:
        print("No data to evaluate. Exiting.")
        return
        
    # Load API metadata
    api_descriptions, api_list = load_api_metadata()
    
    # Evaluate base model
    base_model_results = evaluate_test_set(base_model, test_dataset, api_descriptions, api_list, max_turns)
    
    # Evaluate fine-tuned model
    finetuned_model_results = evaluate_test_set(finetuned_model_path, test_dataset, api_descriptions, api_list, max_turns)
    
    # Prepare comparison results
    comparison_results = {
        "base_model": base_model_results,
        "finetuned_model": finetuned_model_results,
        "comparison": {
            "success_rate_diff": finetuned_model_results["metrics"]["success_rate"] - base_model_results["metrics"]["success_rate"],
            "avg_turns_diff": finetuned_model_results["metrics"]["avg_turns_successful"] - base_model_results["metrics"]["avg_turns_successful"]
        }
    }
    
    # Save results
    output_path = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n=== Comparison Results ===")
    print(f"Base model: {base_model}")
    print(f"  Successful queries: {base_model_results['metrics']['successful_queries']}/{base_model_results['metrics']['total_queries']} ({base_model_results['metrics']['success_rate']*100:.2f}%)")
    print(f"  Average turns for successful queries: {base_model_results['metrics']['avg_turns_successful']:.2f}")
    
    print(f"\nFine-tuned model: {finetuned_model_path}")
    print(f"  Successful queries: {finetuned_model_results['metrics']['successful_queries']}/{finetuned_model_results['metrics']['total_queries']} ({finetuned_model_results['metrics']['success_rate']*100:.2f}%)")
    print(f"  Average turns for successful queries: {finetuned_model_results['metrics']['avg_turns_successful']:.2f}")
    
    print(f"\nSuccess rate improvement: {comparison_results['comparison']['success_rate_diff']*100:.2f}%")
    
    turn_diff = comparison_results['comparison']['avg_turns_diff']
    better_worse = "fewer" if turn_diff < 0 else "more"
    print(f"Average turns difference: {abs(turn_diff):.2f} {better_worse} turns")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)