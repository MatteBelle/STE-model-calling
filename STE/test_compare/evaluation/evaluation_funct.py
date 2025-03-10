"""
Shared evaluation utilities used across the project.
"""

import evaluate
import json
from typing import Dict, Any, List
from config.constants import TEMPERATURE
from STE.my_llm import chat_my

# Export the METRIC_CACHE as a module variable
METRIC_CACHE = {}

def normalize_evaluation_args(metric_name, args, API_descriptions):
    """
    Normalize and coerce the evaluation inputs so that they match the expected types.
    This function uses the metadata for the given metric (from API_descriptions)
    to determine expected types for each parameter.
    """
    try:
        # Load metadata for the metric (the value is a JSON string)
        metric_meta = API_descriptions[metric_name]
    except Exception as e:
        print(f"DEBUG: Error loading metadata for {metric_name}: {e}", flush=True)
        metric_meta = {}

    normalized_args = {}

    # Merge "kwargs" into the main args dictionary if present
    if "kwargs" in args and isinstance(args["kwargs"], dict):
        print("DEBUG: Expanding 'kwargs' into args.", flush=True)
        args.update(args.pop("kwargs"))

    # Get a list of expected parameters from required and optional parameters.
    param_list = metric_meta.get("required_parameters", []) + metric_meta.get("optional_parameters", [])

    # Build a mapping: parameter name -> expected type (as lower-case string)
    expected_types = {param["name"]: param["type"].lower() for param in param_list}

    for key, value in args.items():
        exp_type = expected_types.get(key, None)

        if exp_type is None:
            # If we do not have a type specification, leave the value as is.
            normalized_args[key] = value
        else:
            try:
                if exp_type.startswith("list"):
                    # Expected a list.
                    # Instead of splitting by commas, simply wrap the string in a list.
                    if not isinstance(value, list):
                        if isinstance(value, str):
                            # Do not split by commas; treat the entire string as a single list element.
                            normalized_args[key] = [value.strip()]
                        else:
                            normalized_args[key] = [value]
                    else:
                        normalized_args[key] = value
                elif exp_type == "boolean":
                    # Convert string representations to boolean.
                    if isinstance(value, str):
                        normalized_args[key] = True if value.lower() in ["true", "1", "yes"] else False
                    else:
                        normalized_args[key] = bool(value)
                elif exp_type == "number":
                    # Attempt to convert to integer or float.
                    if isinstance(value, (int, float)):
                        normalized_args[key] = value
                    else:
                        str_val = str(value)
                        if "." in str_val:
                            normalized_args[key] = float(str_val)
                        else:
                            normalized_args[key] = int(str_val)
                elif exp_type == "string":
                    normalized_args[key] = str(value)
                else:
                    normalized_args[key] = value
            except Exception as e:
                print(f"DEBUG: Error normalizing parameter '{key}' with value '{value}': {e}", flush=True)
                normalized_args[key] = value
    return normalized_args

def run_evaluation(metric_name, args, API_list, API_descriptions, truncate=False):
    """
    Execute an evaluation metric from Hugging Face evaluate.
    For the custom metric 'llm_judge', use the LLM to judge text quality.
    """
    global METRIC_CACHE
    if metric_name not in API_list:
        raise ValueError(f"Metric '{metric_name}' is not supported. Supported metrics are: {API_list}")
    
    # First, normalize the arguments.
    try:
        print(f"DEBUG: Normalizing evaluation arguments for metric '{metric_name}'", flush=True)
        print(f"DEBUG: Arguments before normalization: {args}", flush=True)
        normalized_args = normalize_evaluation_args(metric_name, args, API_descriptions)
        print("DEBUG: NORMALIZED ARGS = " + str(normalized_args), flush=True)
    except Exception as e:
        return f"Normalization error: {str(e)}"
    
    # If the metric is our custom llm_judge, handle it separately.
    if metric_name == "llm_judge":
        return run_llm_judge_evaluation(normalized_args, API_descriptions)
    
    # For all other metrics, use the standard evaluate.load mechanism.
    try:
        if metric_name not in METRIC_CACHE:
            METRIC_CACHE[metric_name] = evaluate.load(metric_name)
        metric = METRIC_CACHE[metric_name]
        
        # Special cases: adjust parameters if needed.
        if metric_name == "bertscore":
            print("DEBUG: Overriding model_type for bertscore to 'google/bert_uncased_L-2_H-128_A-2'", flush=True)
            normalized_args["model_type"] = "google/bert_uncased_L-2_H-128_A-2"
        if metric_name == "perplexity":
            print("DEBUG: Overriding model_id for perplexity to 'gpt-2'", flush=True)
            normalized_args["model_id"] = "gpt2"
        
        result = metric.compute(**normalized_args)
        result_str = json.dumps(result)
        evaluated = True
        if truncate:
            result_str = result_str[:truncate]
        return result_str, evaluated
    except Exception as e:
        print("DEBUG ERROR: " + str(e), flush=True)
        return f"The Action or Action Input is incorrect: {str(e)}. Fix it and provide new Action or Action input.", False
    
def run_llm_judge_evaluation(normalized_args, API_descriptions, temp=TEMPERATURE, max_tokens=512):
    """
    Custom handler for the 'llm_judge' metric. This function constructs a prompt
    to use the LLM as a judge for evaluating a candidate text based on multiple quality criteria.

    Expected keys in normalized_args for llm_judge:
        - candidate_texts (LIST of STRING): The text to evaluate.
        - quality_criteria (LIST of STRING): Aspects to consider (e.g., coherence, creativity).
        - scale_max (NUMBER): The top of the evaluation scale.
        - explanation_required (BOOLEAN, optional): Whether to output an explanation.
        - evaluation_type (STRING, optional): e.g., 'numeric' (default) or others.
        - prompt_template (STRING, optional): A custom prompt snippet to guide evaluation.

    Returns:
        JSON string containing the evaluation result (e.g., {"score": 8, "explanation": "..."})
    """
    # Ensure JSON-safe formatting for normalized_args
    json_args = json.dumps(normalized_args, ensure_ascii=False, indent=2)
    # Prepare a basic system message.
    messages = [{"role": "system", "content": "You are an LLM acting as an evaluation function with this documentation: " + json.dumps(API_descriptions["llm_judge"], ensure_ascii=False)}]

    prompt = (
        f"Your input parameters:\n```json\n{json_args}\n```\n\n"
        "You are an LLM Judge evaluating text quality.\n"
        "**Your ONLY task** is to output a JSON response formatted like this:\n\n"
        "```json\n"
        "{\n"
        '  "scores": {\n'
        '    "metric1": [score1, score2, ..., scoreN],\n'
        '    "metric...": [score1, score2, ..., scoreN],\n'
        '    "metricJ": [score1, score2, ..., scoreN]\n'
        "  },\n"
        '  "scale_max": <integer>, # scale_max\n'
        '  "explanation": "<text>"  # Only if explanation_required=true\n'
        "}\n"
        "```\n"
        "**STRICT INSTRUCTIONS:**\n"
        "- Respond ONLY with a JSON object.\n"
        "- Do NOT include any extra text before or after the JSON, except for exactly the text `'Evaluation Ends' right after the json.\n"
    )
    try:
        response, _ = chat_my(messages, prompt, temp=temp, stop="Evaluation Ends", visualize=False, max_tokens=max_tokens)
        response = response[-1]['content']
        evaluated = True
        return response, evaluated
    except Exception as e:
        # If parsing fails, return an error message.
        # return json.dumps({
        #     "error": f"Failed to process llm_judge response: {str(e)}",
        #     "raw_response": response if 'response' in locals() else ""
        # }), False
        return json.dumps({
            "error": f"Failed to process llm_judge response: {str(e)}"
        }), False