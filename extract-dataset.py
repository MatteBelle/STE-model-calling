import json
import os
import glob
import re

def load_api_list(script_dir):
    """Load the API list from the same directory as the script."""
    api_list_path = os.path.join(script_dir, "API_list.json")
    try:
        with open(api_list_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load API list from {api_list_path}: {e}")
        # Fall back to STE directory if not found in script directory
        fallback_path = os.path.join(script_dir, "STE", "tool_metadata", "API_list.json")
        try:
            with open(fallback_path, 'r', encoding='utf-8') as f:
                print(f"Using fallback API list from {fallback_path}")
                return json.load(f)
        except Exception as e2:
            print(f"Warning: Could not load fallback API list from {fallback_path}: {e2}")
            return []

def load_api_descriptions(script_dir):
    """Load API descriptions from the same directory as the script."""
    descriptions_path = os.path.join(script_dir, "API_descriptions.json")
    try:
        with open(descriptions_path, 'r', encoding='utf-8') as f:
            api_descriptions = json.load(f)
            # Remove _meta entry from descriptions
            if "_meta" in api_descriptions:
                del api_descriptions["_meta"]
            return api_descriptions
    except Exception as e:
        print(f"Warning: Could not load API descriptions from {descriptions_path}: {e}")
        # Fall back to STE directory if not found in script directory
        fallback_path = os.path.join(script_dir, "STE", "tool_metadata", "API_descriptions.json")
        try:
            with open(fallback_path, 'r', encoding='utf-8') as f:
                api_descriptions = json.load(f)
                print(f"Using fallback API descriptions from {fallback_path}")
                # Remove _meta entry from descriptions
                if "_meta" in api_descriptions:
                    del api_descriptions["_meta"]
                return api_descriptions
        except Exception as e2:
            print(f"Warning: Could not load fallback API descriptions from {fallback_path}: {e2}")
            return {}

def format_api_description(metric, description_obj):
    """Format all fields from the API description object."""
    formatted = f"Metric name: {metric}\n"
    
    # Add all fields from the description object
    for field, value in description_obj.items():
        # Skip complex objects like examples to keep the text representation clean
        if isinstance(value, (dict, list)) and field != "required_parameters" and field != "optional_parameters":
            continue
            
        if field == "description":
            formatted += f"  Description: {value}\n"
        elif field == "required_parameters":
            formatted += "  Required parameters:\n"
            if isinstance(value, list):
                for param in value:
                    if isinstance(param, dict) and "name" in param:
                        formatted += f"    - {param.get('name', '')}"
                        if "description" in param:
                            formatted += f": {param.get('description', '')}"
                        formatted += "\n"
                    else:
                        formatted += f"    - {param}\n"
            else:
                formatted += f"    {value}\n"
        elif field == "optional_parameters":
            formatted += "  Optional parameters:\n"
            if isinstance(value, list):
                for param in value:
                    if isinstance(param, dict) and "name" in param:
                        formatted += f"    - {param.get('name', '')}"
                        if "description" in param:
                            formatted += f": {param.get('description', '')}"
                        formatted += "\n"
                    else:
                        formatted += f"    - {param}\n"
            else:
                formatted += f"    {value}\n"
        else:
            # Format other simple fields
            formatted += f"  {field.capitalize()}: {value}\n"
    
    return formatted

def find_metrics_in_query(query, api_list):
    """Find all metrics mentioned in the query."""
    found_metrics = []
    for api in api_list:
        # Case insensitive search for the API name in the query
        if re.search(r'\b' + re.escape(api) + r'\b', query, re.IGNORECASE):
            found_metrics.append(api.lower())
    return found_metrics

def process_json_file(file_path, api_list, api_descriptions):
    """Process a JSON file to extract queries and actions where solved_at_turn > 0."""
    try:
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        # Process all metric types in the JSON structure
        for metric_type in data.keys():
            # Skip non-metric keys
            if metric_type in ["ALL_METRICS", "ALL_METRICS_DESCRIPTIONS", "Hyperparameters"]:
                continue
                
            # Process each item group for the metric
            for item_group in data.get(metric_type, []):
                for item in item_group.get("item_list", []):
                    # Check if the item was solved
                    if item.get("solved_at_turn", 0) > 0:
                        # Extract the query
                        query = item.get("query", "")
                        
                        # Find metrics mentioned in the query
                        mentioned_metrics = find_metrics_in_query(query, api_list)
                        
                        # Build documentation string
                        documentation = ""
                        if mentioned_metrics:
                            documentation = "Metrics documentation:\n"
                            for metric in mentioned_metrics:
                                if metric in api_descriptions:
                                    metric_desc = format_api_description(metric, api_descriptions[metric])
                                    documentation += f"{metric_desc}\n"
                        
                        # Combine documentation with query
                        user_query = f"{documentation}\n\n query: {query}"
                        
                        # Find the last non-empty actions list
                        last_actions = []
                        for chain in item.get("chains", []):
                            actions_list = chain.get("actions", [])
                            if actions_list:
                                last_actions = actions_list
                        
                        # Format the actions and action_inputs
                        actions_text = ""
                        if last_actions:
                            for i, action_item in enumerate(last_actions):
                                action = action_item.get("action", "")
                                action_input = action_item.get("action_input", {})
                                actions_text += f"Action: {action}\nAction Input: {json.dumps(action_input, indent=2)}"
                                if i < len(last_actions) - 1:
                                    actions_text += "\n\n"
                        
                        if query and actions_text:  # Only add if both query and actions are present
                            results.append({
                                "query": user_query,
                                "answer": actions_text,
                                "metric_type": metric_type
                            })
        
        return results
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not script_dir:  # If script_dir is empty, use current directory
        script_dir = os.getcwd()
    
    # Load API list and descriptions
    api_list = load_api_list(script_dir)
    api_descriptions = load_api_descriptions(script_dir)
    
    print(f"Loaded {len(api_list)} APIs from API_list.json")
    print(f"Loaded {len(api_descriptions)} API descriptions from API_descriptions.json")
    
    # Find all JSON files in the script directory and STE/saved_results directory
    json_files = glob.glob(os.path.join(script_dir, "*.json"))
    saved_results_dir = os.path.join(script_dir, "STE", "saved_results")
    if os.path.exists(saved_results_dir):
        json_files.extend(glob.glob(os.path.join(saved_results_dir, "*.json")))
    
    all_results = []
    
    for file_path in json_files:
        # Skip API files and dataset.json to avoid processing our metadata or output file
        if os.path.basename(file_path) in ["API_list.json", "API_descriptions.json", "dataset.json"]:
            continue
            
        print(f"Processing {file_path}...")
        results = process_json_file(file_path, api_list, api_descriptions)
        print(f"Found {len(results)} items with solved_at_turn > 0")
        all_results.extend(results)
    
    # Print metrics distribution
    metrics_count = {}
    for item in all_results:
        metrics_count[item['metric_type']] = metrics_count.get(item['metric_type'], 0) + 1
    
    print("\nMetrics distribution:")
    for metric, count in metrics_count.items():
        print(f"  {metric}: {count} items")
    
    # Filter out problematic items
    filtered_results = []
    for item in all_results:
        # Remove metric_type from the item
        del item['metric_type']
        
        # Only keep items that don't contain "is empty" in query or answer
        if ("is empty" not in item['query'].lower() and 
            "the response is empty" not in item['query'].lower() and
            len(item['answer'].strip()) > len("assistant ")):
            filtered_results.append(item)
    
    print(f"Filtered out {len(all_results) - len(filtered_results)} problematic items")
    all_results = filtered_results
    
    # Save the combined results to dataset.json
    output_path = os.path.join(script_dir, "dataset.json")
    print(f"Processing complete. Total items: {len(all_results)}")
    print(f"Results saved to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()