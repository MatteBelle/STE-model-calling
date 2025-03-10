"""
Functions for parsing model responses into actions and input parameters.
"""

import json
from typing import Tuple, Optional, Dict, List


def extract_action_and_input(response: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Extract the first action and action input from model response.
    
    Args:
        response: Model's response text
        
    Returns:
        Tuple of (action, action_input)
    """
    # Use extract_all_actions_and_inputs and return just the first one
    actions = extract_all_actions_and_inputs(response)
    if actions and len(actions) > 0:
        return actions[0]["action"], actions[0]["action_input"]
    return None, None

def parse_response(response, API_name_list, api_descriptions,
                   proc_thought=False, proc_toolken=False, check_API_name=True, ground_API=False):
    item = dict()
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
        item['finish'] = True  # Make sure this is set to True
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
                item['parse_error_msg'] = ("Please only use exactly the following APIs: {}."
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
            item['parse_error_msg'] = ("One Action Input is not in valid JSON format. It should begin with '{' and end with '}'.")
            return item
        json_str = candidate[left_index:right_index+1]
        # Check for extra braces.
        if json_str.startswith("{{") or json_str.endswith("}}"):
            item['parse_successful'] = False
            item['parse_error_msg'] = ("One Action Input should begin with a single '{' and end with a single '}'.")
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