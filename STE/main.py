import json
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import fire
from transformers import pipeline  # Import Hugging Face pipeline
from utils import find_reverse, random_choose, parse_response, strip_end, delete_intermediate_subfolder, trim_ltm_stm, get_metric_subgroup, get_parameters_optionality, build_optional_parameters_text
from my_llm import chat_my, visualize_messages, get_chat_completion_my#, set_tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime  # For generating dynamic filenames
import evaluate
import random

METRIC_CACHE = {}
TEMPERATURE = 0.4

def main(
    num_sessions: int = 4,
    num_stm_slots: int = 2,
    max_turn: int = 3,
    intermediate_dir_write: str = "STE/results/intermediate_results/",
    final_dir_write: str = "STE/results/final_results/",
    if_visualize: bool = True,
):
    placeholder = "[...]"  # used for trimmed LTM lists
    data_dict = dict()

    os.makedirs(final_dir_write, exist_ok=True)
    os.makedirs(intermediate_dir_write, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_path = os.path.join(intermediate_dir_write, f"run_{run_timestamp}")
    os.makedirs(subfolder_path, exist_ok=True)
    subfolder_info_path = os.path.join(intermediate_dir_write, "latest_run_subfolder.txt")
    with open(subfolder_info_path, "w") as f:
        f.write(subfolder_path)
        
    print(f"DEBUG: Intermediate results will be stored in: {subfolder_path}", flush = True)
    print("DEBUG: Entering main function with parameters:", flush=True)
    print(f"num_sessions={num_sessions}, num_stm_slots={num_stm_slots}, max_turn={max_turn}, final_dir_write={final_dir_write}, if_visualize={if_visualize}", flush=True)
    # ... [setup and directory creation as before]

    # Load API descriptions and API list.
    with open("STE/tool_metadata/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)
    with open("STE/tool_metadata/API_list.json", "r", encoding='utf-8') as f:
        API_list = json.load(f)
    data_dict["ALL_METRICS"] = API_list
    data_dict["ALL_METRICS_DESCRIPTIONS"] = API_descriptions

    # Memory and reflection prompts remain unchanged.
    PAST_Q_MSG_pre = "Below are queries you have already explored and whether you successfully solved them with the API's help:"
    PAST_Q_MSG_post = f"Based on these, try to explore queries that can help you understand the API further; avoid synthesizing queries that are too close to the existing ones and remember that {placeholder} is a placeholder I use for trimmed texts, so don't use {placeholder} it in your queries. Try different ways to formulate the query (use synonyms, vary its structure, vary the length of the question, change the references parameters etc.)."
    prompt_reflection = "Do you think you successfully fulfilled this query in the end? Respond with \"Yes\" or \"No\"."

    print("DEBUG: Starting iteration over API_list.", flush=True)
    for API in API_list:
        print("DEBUG: Processing API:", API, flush=True)
        with open("STE/prompts/prompt_explore.txt", "r") as f:
            prompt_template = f.read().strip()

        template_q, template_a, template_q_follow, template_a_follow = prompt_template.split("=========")
        template_q, template_a, template_q_follow, template_a_follow = template_q.strip(), template_a.strip(), template_q_follow.strip(), template_a_follow.strip()
        #template_q_follow = template_q_follow.format(placeholder=placeholder) commented as it's later formatted with optionality as well.
        all_sessions, explored_queries, whether_successful = [], [], []
        for session_id in range(num_sessions):
            print("DEBUG: Starting episode:", session_id, flush = True)    
            # Extract subgroups that includes (API) and parameters optionality    
            subgroup = get_metric_subgroup(API)
            API_name_list = subgroup["metrics"]
            optional_flags = get_parameters_optionality(API_name_list)
            
            item_list = []
            first_item = dict()
            api_descriptions_text = "\n\n".join(["API_name: {}\nDescription: {}".format(temp, API_descriptions[temp]) for temp in API_name_list])
            optional_parameters_text = build_optional_parameters_text(
                API_name_list,
                optional_flags,
                API_descriptions
            )
            # Prompt for query formatted
            prompt_q = template_q.format(
                api_descriptions=api_descriptions_text,
                optional_parameters=optional_parameters_text
            )

            if len(explored_queries) > 0:
                try:
                    assert prompt_q.endswith("User Query:")
                except AssertionError:
                    print(f"DEBUG: Assertion failed! prompt_q does not end with 'User Query:'. Tail of prompt_q: {prompt_q[-20:]}", flush=True)
                    raise
                prompt_q_added_question = strip_end(prompt_q, "User Query:").strip()
                prompt_q_added_question = prompt_q_added_question + "\n\n" + \
                    PAST_Q_MSG_pre + "\n" + "\n".join(trim_ltm_stm(LTM(explored_queries, whether_successful), placeholder=placeholder)) + \
                    "\n\n" + PAST_Q_MSG_post + "\n\nUser Query:"
            else:
                prompt_q_added_question = prompt_q

            messages = [
                {"role": "system", "content": "You are a bot that creates and responds to evaluation queries."}
            ]

            print("DEBUG: Generating the first query using chat_my.", flush=True)
            response = chat_my(messages, prompt_q_added_question,
                               temp=TEMPERATURE, stop="Thought:", visualize=if_visualize, max_tokens=512)[-1]['content']

            messages = messages + [
                {"role": "user", "content": prompt_q},
                {"role": "assistant", "content": response}
            ]
            print("DEBUG FIRST USER QUERY OF SESSION " + str(session_id) +  ": \n" + prompt_q_added_question+"\n\n\n\n", flush=True)
            print("DEBUG FIRST RESPONSE OF SESSION " + str(session_id) +  ": \n" + response, flush=True)
            print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)

            query = messages[-1]['content']
            prompt_a = template_a.format(
                metric_name=", ".join(API_name_list),
                query=query,
            )
            first_item['query'] = query
            explored_queries.append(query)

            chains = []
            print("DEBUG: Processing chain of calls for the first query.", flush=True)
            messages = chat_my(messages, prompt_a, temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=512)            
            temp = messages[-1]['content']

            print("DEBUG FIRST USER ANSWER OF SESSION " + str(session_id) +  ": \n" + messages[-2]['content'] +"\n\n\n\n", flush=True)
            print("DEBUG FIRST ACTION AND INPUT OF SESSION " + str(session_id) +  ": \n" + temp, flush=True)
            print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
            
            parsed_response = parse_response(temp, API_name_list, API_descriptions)
            for n_turn in range(max_turn):
                if not parsed_response['parse_successful']:
                    evaluation_result = parsed_response['parse_error_msg']
                else:
                    if parsed_response.get('finish', False):
                        print("DEBUG: Final answer reached. Moving to next episode.", flush=True)
                        chains.append(parsed_response)
                        break
                    else:
                        evaluation_result = ""
                        # Loop over each extracted action and call run_evaluation.
                        for act in parsed_response['actions']:
                            evaluation_result += run_evaluation(act['action'], act['action_input'], API_list, API_descriptions) + "\n"
                parsed_response['evaluation_result'] = evaluation_result
                chains.append(parsed_response)

                messages = chat_my(messages, 'Evaluation Result: ' + evaluation_result,
                                   temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=512)
                temp = messages[-1]['content']
                print("DEBUG USER QUERY OF SESSION " + str(session_id) + ", TURN " + str(n_turn) + ", PARSING ACTION AND INPUT: \n" + messages[-2]['content'] + "\n\n\n\n", flush=True)
                print("DEBUG RESPONSE OF SESSION " + str(session_id) + ", TURN " + str(n_turn) + ", PARSING ACTION AND INPUT: \n" + temp, flush=True)
                print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
                
                parsed_response = parse_response(temp, API_name_list, API_descriptions)
            if len(chains) == 0 or not parsed_response.get('finish', False):
                chains.append(parsed_response)

            first_item['chains'] = chains

            print("DEBUG: Running reflection to determine success for the first query.", flush=True)
            messages = chat_my(messages, prompt_reflection, temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=512)
            res = messages[-1]['content']
            print("DEBUG USER QUERY REFLECTION OF SESSION " + str(session_id) + ": \n" + prompt_reflection + "\n\n\n\n", flush=True)
            print("DEBUG RESPONSE OF SESSION " + str(session_id) + ": \n" + res, flush=True)
            print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
            
            if "No" in res:
                successful = "No"
                first_item["solved_at_turn"] = -1  # Set to -1 if never solved
            else:
                successful = "Yes"
                first_item['solved_at_turn'] = n_turn + 1  # Turn count starts at 1

            whether_successful.append(successful)
            item_list.append(first_item)

            # Process additional STM slots similarly.
            for n_stm in range(num_stm_slots-1):
                item = dict()
                new_optional_flags = get_parameters_optionality(API_name_list)
                optional_parameters_text = build_optional_parameters_text(API_name_list,new_optional_flags,API_descriptions)
                # Prompt for query formatted
                template_q_follow = template_q_follow.format(
                    placeholder = placeholder,
                    optional_parameters=build_optional_parameters_text(API_name_list,new_optional_flags,API_descriptions)
                )
                    
                if len(explored_queries) > 0:
                    try:
                        assert template_q_follow.endswith("User Query:")
                    except AssertionError:
                        print(f"DEBUG: Assertion failed! prompt_q_follow does not end with 'User Query:'. Tail of prompt_q_follow: {template_q_follow[-20:]}", flush=True)
                        raise
                    template_q_follow_added_question = strip_end(template_q_follow, "User Query:").strip()
                    template_q_follow_added_question = template_q_follow_added_question + "\n\n" + \
                    PAST_Q_MSG_pre + "\n" + "\n".join(trim_ltm_stm(LTM(explored_queries, whether_successful), placeholder=placeholder)) + \
                    "\n\n" + PAST_Q_MSG_post + "\n\nUser Query:"
                else:
                    template_q_follow_added_question = template_q_follow

                print("DEBUG: Generating follow-up query using chat_my.", flush=True)
                response = chat_my(messages, template_q_follow_added_question,
                                   temp=TEMPERATURE, stop="Thought:", visualize=if_visualize, max_tokens=512)[-1]['content']
                messages = messages + [
                    {"role": "user", "content": template_q_follow_added_question},
                    {"role": "assistant", "content": response}
                ]
                print("DEBUG USER QUERY REFLECTION OF SESSION " + str(session_id) + ", STM TURN " + str(n_stm) + ": \n" + template_q_follow_added_question + "\n\n\n\n", flush=True)
                print("DEBUG RESPONSE OF SESSION " + str(session_id) + ", STM TURN " + str(n_stm) + ": \n" + response, flush=True)
                print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
                query = messages[-1]['content']
                item['query'] = query
                explored_queries.append(query)

                chains = []
                print("DEBUG: Processing chain of calls for the short-term memory slot query.", flush=True)
                messages = chat_my(messages, template_a_follow,
                                   temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=512)
                temp = messages[-1]['content']
                print("DEBUG template_a_follow OF SESSION " + str(session_id) + ", STM TURN " + str(n_stm) + ": \n" + template_a_follow + "\n\n\n\n", flush=True)
                print("DEBUG RESPONSE OF SESSION " + str(session_id) + ", STM TURN " + str(n_stm) + ": \n" + temp, flush=True)
                print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
                parsed_response = parse_response(temp, API_name_list, API_descriptions)
                for n_turn in range(max_turn):
                    if not parsed_response['parse_successful']:
                        evaluation_result = parsed_response['parse_error_msg']
                    else:
                        if parsed_response.get('finish', False):
                            chains.append(parsed_response)
                            break
                        else:
                            evaluation_result = ""
                            for act in parsed_response['actions']:
                                evaluation_result += run_evaluation(act['action'], act['action_input'], API_list, API_descriptions) + "\n"
                    parsed_response['evaluation_result'] = evaluation_result
                    chains.append(parsed_response)

                    messages = chat_my(messages, 'Evaluation Result: ' + evaluation_result,
                                       temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=512)
                    temp = messages[-1]['content']
                    print("DEBUG USER QUERY OF SESSION " + str(session_id) + ", TURN " + str(n_turn) + ", PARSING ACTION AND INPUT: \n" + messages[-2]['content'] + "\n\n\n\n", flush=True)
                    print("DEBUG RESPONSE OF SESSION " + str(session_id) + ", TURN " + str(n_turn) + ", PARSING ACTION AND INPUT: \n" + temp, flush=True)
                    print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
                    parsed_response = parse_response(temp, API_name_list, API_descriptions)
                if len(chains) == 0 or not parsed_response.get('finish', False):
                    chains.append(parsed_response)

                item['chains'] = chains

                print("DEBUG: Running reflection to determine success for the short-term memory slot query.", flush=True)
                messages = chat_my(messages, prompt_reflection,
                                   temp=TEMPERATURE, stop="Evaluation Result:", visualize=if_visualize, max_tokens=512)
                res = messages[-1]['content']
                print("DEBUG USER QUERY REFLECTION OF SESSION " + str(session_id) + ", TURN " + str(n_turn) + ": \n" + messages[-2]['content'] + "\n\n\n\n", flush=True)
                print("DEBUG RESPONSE OF SESSION " + str(session_id) + ", TURN " + str(n_turn) + ": \n" + res, flush=True)
                print("DEBUG END: --------------------------------------------------------------------------------------------------------------\n\n\n\n\n\n\n\n", flush=True)
                if "No" in res:
                    successful = "No"
                    item['solved_at_turn'] = -1
                else:
                    successful = "Yes"
                    item['solved_at_turn'] = n_turn + 1  # Turn count starts at 1
                whether_successful.append(successful)
                item_list.append(item)

            all_sessions.append(
                {
                    "item_list": item_list,
                    "messages": messages,
                }
            )
        save_intermediate_results(API, 99, all_sessions, subfolder_path)
        data_dict[API] = all_sessions
    final_data_path = os.path.join(final_dir_write, f"data_dict_{run_timestamp}.json")
    data_dict = sanitize_for_json(data_dict)
    with open(final_data_path, "w", encoding='utf-8') as f:
        json.dump(data_dict, f)
    delete_intermediate_subfolder(subfolder_path)
    print(f"DEBUG: Final data saved to {final_data_path}", flush=True)
    print("DEBUG: Finished main function.", flush=True)


# def LTM(X, labels):
#     #assert len(X) == len(labels)
#     try:
#         assert len(X) == len(labels)
#     except AssertionError:
#         print(f"DEBUG: Assertion failed! len(X, flush=True) == len(labels) '. len(X) {len(X)}, len(labels) {len(labels)}")
#         raise  # Re-raise the exception after logging
#     print("DEBUG: LTMLTM: " + str(["Query: {} \n Solved: {}".format(X[i], labels[i], flush=True) for i in range(len(X))]))
#     return ["Query: {} \n Solved: {}".format(X[i], labels[i]) for i in range(len(X))]

def LTM(X, labels, max_items=10):
    """
    Creates a formatted list of past queries and their success status.
    If the list exceeds max_items, it will keep the most recent ones.
    
    Args:
        X: List of queries
        labels: List of success labels
        max_items: Maximum number of items to include
    
    Returns:
        List of formatted strings
    """
    try:
        assert len(X) == len(labels)
    except AssertionError:
        print(f"DEBUG: Assertion failed! len(X) == len(labels) '. len(X) {len(X)}, len(labels) {len(labels)}")
        raise
    
    # If the list is too long, keep only the most recent items
    if len(X) > max_items:
        X = X[-max_items:]
        labels = labels[-max_items:]
    
    formatted_list = ["Query: {} \n Solved: {}".format(X[i], labels[i]) for i in range(len(X))]
    print("DEBUG: LTMLTM: " + str(formatted_list))
    return formatted_list

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
        if truncate:
            result_str = result_str[:truncate]
        return result_str
    except Exception as e:
        print("DEBUG ERROR: " + str(e), flush=True)
        return f"The Action or Action Input is incorrect: {str(e)}. Fix it and provide new Action or Action input."

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
        response = chat_my(messages, prompt, temp=temp, stop="Evaluation Ends", visualize=False, max_tokens=max_tokens)[-1]['content']
        return response
    except Exception as e:
        # If parsing fails, return an error message.
        return json.dumps({
            "error": f"Failed to process llm_judge response: {str(e)}",
            "raw_response": response if 'response' in locals() else ""
        })

def save_intermediate_results(API, session_id, all_sessions, subfolder_path):
    """
    Saves intermediate results in the dynamically created subfolder inside intermediate_dir_write.
    """
    try:
        # Generate a timestamp for each session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        intermediate_filename = os.path.join(subfolder_path, f"intermediate_{API}_session_{session_id}_{timestamp}.json")

        with open(intermediate_filename, "w", encoding="utf-8") as f:
            json.dump({
                "API": API,
                "session_id": session_id,
                "all_sessions_so_far": all_sessions
            }, f, ensure_ascii=False, indent=2)

        print(f"DEBUG: Intermediate results saved to {intermediate_filename}", flush=True)
    except Exception as e:
        print(f"DEBUG: Error saving intermediate results: {e}", flush=True)

def sanitize_for_json(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    return obj

if __name__ == '__main__':
    fire.Fire(main)
    