import json
import random
import os


def get_parameters_optionality(metrics):
    """
    For each metric in 'metrics', assign a boolean flag 
    that is True with 30% probability (False otherwise).

    Returns:
        dict: { metric_name: bool }
    """
    optional_flags = {}
    for metric in metrics:
        optional_flags[metric] = (random.random() < 0.6)
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
    lines.append("**Parameters Mandatory Instructions**")

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
                f"- For {metric}: You MUST include these optional parameters in your user query: {', '.join(param_names)}."
            )
        else:
            lines.append(
                f"- For {metric}: Do NOT include any optional parameters in your user query."
            )
            
    lines.append("In case optional parameters ara asked to be included, the query must explicitly specify both the parameters and a chosen value.")
    # Join everything into a single text block
    return "\n".join(lines)

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

    
if __name__ == '__main__':
    print()