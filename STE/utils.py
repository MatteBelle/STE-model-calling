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
    # Commented these as I don't want all api names and descriptions to be repeated, I added them just once in the beginning
    # item['API_name_list'] = API_name_list
    # item['api_descriptions'] = api_descriptions

    item['parse_successful'] = True

    # NEW: Clean up the chat template special tokens from the response.
    # This removes tokens introduced by apply_chat_template.
    for token in ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
        response = response.replace(token, "")
    # Also remove common role labels if they appear (in case they are present without the special tokens)
    for role in ["system", "user", "assistant"]:
        response = response.replace(role, "")
    response = response.strip()

    if "Final Answer:" in response:
        temp = response.split("Final Answer:")
        response, final_ans = temp[0].strip(), temp[1].strip()
        if "Action Input:" not in response:
            item['final_ans'] = final_ans
            item['finish'] = True
            return item

    item['finish'] = False
    if "Action Input:" not in response:
        item['parse_successful'] = False
        item['parse_error_msg'] = ("If you have already got enough information for the final answer, say "
                                   "\"Final Answer:\" followed by your answer and your consideration on the result and whether other metrics for evaluation could be useful.. Otherwise, please specify your API call via "
                                   "\"Action:\" and API arguments via \"Action Input:\" followed by a json string. "
                                   "If there are no arguments, use \"Action Input: {}\". Do NOT start your response with "
                                   "\"Evaluation Result:\"; there is no need to repeat it.")
        return item

    if response.count("Action Input:") > 1:
        item['parse_successful'] = False
        item['parse_error_msg'] = "Please use only one \"Action Input:\" in your response."
        return item

    action, action_input = response.split("Action Input:")
    action, action_input = strip_end(action.strip(), "\\n").strip(), strip_end(action_input.strip(), "\\n").strip()

    # get action
    if "Action:" not in action:
        item['parse_successful'] = False
        item['parse_error_msg'] = ("Please specify the API name you would like to call via \"Action:\" followed by the name. "
                                   "Remember that you should only call one API at a time, and the API name should be one of the following: {}. "
                                   "If you have already got the final answer, say \"Final Answer:\" followed by your final answer.").format(
                                    ", ".join(API_name_list))
        return item

    if action.count("Action:") > 1:
        item['parse_successful'] = False
        item['parse_error_msg'] = "Please use only one \"Action:\" in your response."
        return item

    thought, action = action.split("Action:")
    thought, action = strip_end(thought.strip(), "\\n").strip(), strip_end(action.strip(), "\\n").strip()

    if proc_toolken:
        action = action.replace("<tool_", "").strip("<>")

    if check_API_name and (action not in API_name_list):
        if ground_API:
            # find the closest API that is supported
            action = get_close_matches(action, API_name_list, n=1, cutoff=0.001)[0]
        else:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Please only use exactly one of the following APIs: {}.".format(
                ", ".join(API_name_list))
            return item

    if proc_thought:
        if "Thought:" not in thought:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Your thought should begin with \"Thought:\"."
            return item

        if thought.count("Thought:") > 1:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Please use only one \"Thought:\" in your response."
            return item

        thought = thought.split("Thought:")[-1].strip()

    # get action input
    left_bracket_pos = action_input.find('{')
    if left_bracket_pos == -1:
        item['parse_successful'] = False
        item['parse_error_msg'] = "the Action Input is in json string format, and should begin with \"{\""
        return item
    right_bracket_pos = find_reverse(action_input, '}')
    if right_bracket_pos == -1:
        item['parse_successful'] = False
        item['parse_error_msg'] = "the Action Input is in json string format, and should end with \"}\". Do NOT say anything else after \"}\""
        return item

    if left_bracket_pos >= right_bracket_pos:
        item['parse_successful'] = False
        item['parse_error_msg'] = "Your action input cannot be parsed as a json string. Please try again."
        return item

    # keep only within {}
    action_input = action_input[left_bracket_pos: right_bracket_pos + 1]
    action_input = "{" + action_input.strip("{}") + "}"

    if action_input.startswith("{{"):
        item['parse_successful'] = False
        item['parse_error_msg'] = "the Action Input is in json string format, and should begin with only one \"{\", not two or more."
        return item
    if action_input.endswith("}}"):
        item['parse_successful'] = False
        item['parse_error_msg'] = "the Action Input is in json string format, and should end with only one \"}\". Do NOT say anything else after \"}\""
        return item

    action_input = action_input.strip()

    # Convert the JSON string into a Python dictionary
    try:
        print("DEBUG: BEFORE JSONLOADS: ", action_input)
        action_input_obj = json.loads(action_input)
    except Exception as e:
        item['parse_successful'] = False
        item['parse_error_msg'] = "Error parsing JSON in Action Input: " + str(e)
        return item

    print("DEBUG: ACTION INPUT object after JSON conversion:", action_input_obj)
    print("DEBUG: ACTION INPUT type after JSON conversion:", type(action_input_obj))

    item['parse_successful'] = True
    if proc_thought:
        item['thought'] = thought
    item['action'] = action
    item['action_input'] = action_input_obj
    return item

def get_random_metric_subgroup_with_flags(json_path="tool_metadata/API_subgroups.json"):
    """
    Loads the metric subgroups from a JSON file, randomly selects one subgroup,
    and for each metric in that subgroup, generates a boolean flag that is True
    with 30% probability (and False with 70% probability).

    Parameters:
        json_path (str): Path to the JSON file containing the metric subgroups.

    Returns:
        dict: A dictionary with the following keys:
            - "name": The name of the selected subgroup.
            - "metrics": A list of metric names in the subgroup.
            - "optional_flags": A dictionary mapping each metric name to a boolean flag.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Subgroups JSON file not found at: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        subgroups = json.load(f)
    
    # Randomly choose one subgroup from the dictionary values.
    chosen_subgroup = random.choice(list(subgroups.values()))
    
    # For each metric in the subgroup, assign a boolean flag (True with probability 0.3).
    optional_flags = {metric: (random.random() < 0.3) for metric in chosen_subgroup["metrics"]}
    
    return {
        "name": chosen_subgroup["name"],
        "metrics": chosen_subgroup["metrics"],
        "optional_flags": optional_flags
    }

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