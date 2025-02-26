import numpy as np
import json
from difflib import get_close_matches
import random
import os
import re

# def clean_response(response):
# # Remove lines that start with “user:”, “assistant:”, or “system:”
#     return re.sub(r’^(user|assistant|system)\s*’, ‘’, response, flags=re.IGNORECASE | re.MULTILINE)
# response = clean_response(response)

def find_reverse(str_a, ch):
    assert type(str_a) == type(ch) == str
    for i in range(len(str_a) - 1, -1, -1):
        if str_a[i] == ch:
            return i
    return -1


def random_choose(l, num):
    if len(l) <= num:
        return l
    inds = np.random.choice(len(l), num, replace=False).tolist()
    return [l[i] for i in inds]


def strip_end(a, b):
    while a.endswith(b):
        a = a[:len(a) - len(b)]
    return a


def parse_response(response, API_name_list, api_descriptions,
                   proc_thought=False, proc_toolken=False, check_API_name=True, ground_API=False):
    item = dict()
    # (Optionally include API metadata only once if needed)
    # item['API_name_list'] = API_name_list
    # item['api_descriptions'] = api_descriptions

    item['parse_successful'] = True
    item['actions'] = []

    # Clean up special tokens and common role labels.
    for token in ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
        response = response.replace(token, "")
    for role in ["system", "user", "assistant"]:
        response = response.replace(role, "")
    response = response.strip()

    # Check for final answer; if found and no actions precede, mark finish.
    if "Final Answer:" in response:
        parts = response.split("Final Answer:")
        pre_final = parts[0].strip()
        final_ans = parts[-1].strip()
        # If no Action Input is found before Final Answer, then we assume the process is complete.
        if "Action Input:" not in pre_final:
            item['final_ans'] = final_ans
            item['finish'] = True
            return item
        # Otherwise, we record the final answer and continue to extract actions.
        item['final_ans'] = final_ans
        item['finish'] = True
    else:
        item['finish'] = False

    # Remove the Final Answer part from further processing.
    if "Final Answer:" in response:
        response = response.split("Final Answer:")[0]

    # Now extract all occurrences of Action and corresponding Action Input.
    # We split by "Action:" and ignore the part before the first occurrence.
    raw_actions = response.split("Action:")[1:]
    if not raw_actions:
        item['parse_successful'] = False
        item['parse_error_msg'] = ("No 'Action:' found in your response. Please specify at least one Action and its Action Input "
                                   "using the format:\nAction: <metric name>\nAction Input: <JSON string>")
        return item

    actions_list = []
    for raw in raw_actions:
        # Each raw block should contain an "Action Input:" token.
        if "Action Input:" not in raw:
            continue  # Skip any malformed block.
        action_text, remainder = raw.split("Action Input:", 1)
        action = action_text.strip()
        if proc_toolken:
            action = action.replace("<tool_", "").strip("<>")
        if check_API_name and (action not in API_name_list):
            if ground_API:
                from difflib import get_close_matches
                action = get_close_matches(action, API_name_list, n=1, cutoff=0.001)[0]
            else:
                item['parse_successful'] = False
                item['parse_error_msg'] = ("Please only use exactly one of the following APIs: {}."
                                           .format(", ".join(API_name_list)))
                return item

        # Extract the JSON block from the remainder.
        # If an "Evaluation Result:" token exists, we take the text before it.
        if "Evaluation Result:" in remainder:
            candidate = remainder.split("Evaluation Result:")[0].strip()
        else:
            candidate = remainder.strip()
        left_index = candidate.find("{")
        right_index = candidate.rfind("}")
        if left_index == -1 or right_index == -1 or right_index <= left_index:
            item['parse_successful'] = False
            item['parse_error_msg'] = ("The Action Input is not in valid JSON format. It should begin with '{' and end with '}'.")
            return item
        json_str = candidate[left_index:right_index+1]
        # Check for extra braces.
        if json_str.startswith("{{") or json_str.endswith("}}"):
            item['parse_successful'] = False
            item['parse_error_msg'] = ("The Action Input should begin with a single '{' and end with a single '}'.")
            return item
        try:
            action_input_obj = json.loads(json_str)
        except Exception as e:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Error parsing JSON in Action Input: " + str(e)
            return item

        actions_list.append({"action": action, "action_input": action_input_obj})
    if not actions_list:
        item['parse_successful'] = False
        item['parse_error_msg'] = ("No valid Action and Action Input pairs were found. Please ensure you follow the format correctly.")
        return item

    item['actions'] = actions_list
    # Optionally, process Thought if proc_thought is True.
    if proc_thought:
        if "Thought:" in response:
            # Get the last occurrence.
            thought = response.split("Thought:")[-1].strip()
            item['thought'] = thought
        else:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Your response should include a 'Thought:' section."
            return item

    return item

def get_metric_subgroup(metric_name: str = None, json_path: str = "STE/tool_metadata/API_subgroups.json"):
    """
    Loads the subgroups from the given JSON file and returns a single subgroup 
    with 'name' and 'metrics'. If 'metric_name' is provided, only subgroups 
    containing that metric will be considered.
    
    Returns:
        dict: {
            "name": <subgroup_name>,
            "metrics": <list_of_metrics_in_that_subgroup>
        }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Subgroups JSON file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        subgroups = json.load(f)

    # Convert dict of subgroups to a list
    subgroup_list = list(subgroups.values())

    if metric_name:
        # Filter subgroups to only those containing the given metric
        filtered_subgroups = [
            subgroup for subgroup in subgroup_list
            if metric_name in subgroup["metrics"]
        ]
        if not filtered_subgroups:
            raise ValueError(
                f"No subgroups found containing the metric '{metric_name}'. "
                f"Available subgroups: {[sg['name'] for sg in subgroup_list]}"
            )
        chosen_subgroup = random.choice(filtered_subgroups)
    else:
        chosen_subgroup = random.choice(subgroup_list)

    return {
        "name": chosen_subgroup["name"],
        "metrics": chosen_subgroup["metrics"]
    }
    
def get_parameters_optionality(metrics):
    """
    For each metric in 'metrics', assign a boolean flag 
    that is True with 30% probability (False otherwise).

    Returns:
        dict: { metric_name: bool }
    """
    optional_flags = {}
    for metric in metrics:
        optional_flags[metric] = (random.random() < 0.3)
    return optional_flags

def get_random_metric_subgroup_with_flags(
    metric_name: str = None, 
    json_path: str = "STE/tool_metadata/API_subgroups.json"
):
    """
    Wrapper that:
      1) Retrieves a subgroup via get_metric_subgroup.
      2) Retrieves parameters optionality flags via get_parameters_optionality.
      3) Combines them into a single dictionary.
    """
    subgroup = get_metric_subgroup(metric_name, json_path)
    optional_flags = get_parameters_optionality(subgroup["metrics"])

    return {
        "name": subgroup["name"],
        "metrics": subgroup["metrics"],
        "optional_flags": optional_flags
    }

def build_optional_parameters_text(API_name_list, optional_flags, API_descriptions):
    """
    Create a guidance text describing whether (True) or not (False)
    to include optional parameters for each metric in the user query.
    """
    lines = []
    lines.append("**Optional Parameters Guidance**")

    for metric in API_name_list:
        # If the metric is missing from optional_flags, skip or default to False
        flag = optional_flags.get(metric, False)
        
        # Get the optional parameters from API_descriptions
        metric_info = API_descriptions.get(metric, {})
        optional_params = metric_info.get("optional_parameters", [])

        # If no optional parameters exist, skip the instruction
        if not optional_params:
            lines.append(f"- For {metric}: (No optional parameters available)")
            continue

        # Build a readable list of optional param names
        param_names = [p["name"] for p in optional_params]

        if flag:
            lines.append(
                f"- For {metric}: Please consider including these optional parameters in your user query: {', '.join(param_names)}."
            )
        else:
            lines.append(
                f"- For {metric}: Do NOT include any optional parameters in your user query."
            )

    # Join everything into a single text block
    return "\n".join(lines)

def delete_intermediate_subfolder(subfolder_path):
    """
    Deletes the entire subfolder and all its contents using os.
    """
    try:
        if os.path.exists(subfolder_path):
            # Walk through the directory tree and delete files and subdirectories
            for root, dirs, files in os.walk(subfolder_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)  # Delete file
                for directory in dirs:
                    dir_path = os.path.join(root, directory)
                    os.rmdir(dir_path)  # Delete empty directory
            os.rmdir(subfolder_path)  # Finally, delete the main folder
            print(f"DEBUG: Successfully deleted intermediate subfolder: {subfolder_path}")
        else:
            print(f"DEBUG: Subfolder does not exist: {subfolder_path}")
    except Exception as e:
        print(f"DEBUG: Error deleting intermediate subfolder: {e}")

def trim_ltm_stm(ltm_list, placeholder="[...]", max_length=2000, list_trim_threshold=40):
    """
    Trims long text lists in the LTM entries and replaces them with a placeholder, 
    while preserving some examples for context. List of lists are trimmed only if too long.

    Args:
        ltm_list (list): List of strings containing the LTM queries.
        placeholder (str): Placeholder text to replace long lists.
        max_length (int): Max length of entire entry before truncation.
        list_trim_threshold (int): Maximum number of characters a list can have before being trimmed.

    Returns:
        list: Trimmed LTM list.
    """
    trimmed_ltm = []

    for entry in ltm_list:
        # Function to selectively trim long lists but keep part of the data
        def trim_list(match):
            content = match.group(0)  # The entire matched list
            if len(content) > list_trim_threshold:
                # Preserve the first few elements of the list while trimming the rest
                partial_content = re.sub(r"(\[.*?, .*?)\s*,\s*.*?\]", rf"\1, {placeholder} ]", content, flags=re.DOTALL)
                return partial_content.replace(f"[{placeholder},", f"[{placeholder}")  # Cleanup format
            return content  # If the list is short, return as is

        # Function to handle nested lists (list of lists)
        def trim_nested_list(match):
            content = match.group(0)  # The entire matched nested list
            if len(content) > list_trim_threshold:
                # Preserve some nested elements while trimming the rest
                partial_content = re.sub(r"(\[\[.*?\], \[.*?\])\s*,\s*.*?\]", rf"\1, {placeholder} ]", content, flags=re.DOTALL)
                return partial_content.replace(f"[{placeholder},", f"[{placeholder}")  # Cleanup format
            return content  # If the nested list is short, return as is

        # Trim only long nested lists (list of lists)
        trimmed_entry = re.sub(r"\[\s*(?:\[[^\]]*\]\s*,?\s*)+\]", trim_nested_list, entry, flags=re.DOTALL)  

        # Trim only long lists inside brackets
        trimmed_entry = re.sub(r"\[.*?\]", trim_list, trimmed_entry, flags=re.DOTALL)  

        # If the whole entry is still too long, truncate the text itself
        if len(trimmed_entry) > max_length:
            trimmed_entry = trimmed_entry[:max_length] + " " + placeholder

        trimmed_ltm.append(trimmed_entry)

    return trimmed_ltm
    
if __name__ == '__main__':
    print()