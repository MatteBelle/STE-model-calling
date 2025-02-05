import json
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import fire
from transformers import pipeline  # Import Hugging Face pipeline
from utils import find_reverse, random_choose, parse_response, strip_end
from my_llm import chat_my, visualize_messages, get_chat_completion_my, set_tokenizer
# Load the LLaMA 2 model and tokenizer, using the cache
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime  # For generating dynamic filenames
import evaluate

def LTM(X, labels):
    print("DEBUG: Calling LTM function.")
    assert len(X) == len(labels)
    return ["Query: {} | Solved: {}".format(X[i], labels[i]) for i in range(len(X))]

def normalize_evaluation_args(metric_name, args, API_descriptions):
    """
    Normalize and coerce the evaluation inputs so that they match the expected types.
    This function uses the metadata for the given metric (from API_descriptions)
    to determine expected types for each parameter.
    """
    try:
        # Load metadata for the metric (the value is a JSON string)
        metric_meta = json.loads(API_descriptions[metric_name])
    except Exception as e:
        print(f"DEBUG: Error loading metadata for {metric_name}: {e}")
        metric_meta = {}
    normalized_args = {}
    # Get a list of expected parameters from required and optional parameters.
    param_list = metric_meta.get("required_parameters", []) + metric_meta.get("optional_parameters", [])
    # Build a mapping: parameter name -> expected type (as lower-case string)
    expected_types = {}
    for param in param_list:
        expected_types[param["name"]] = param["type"].lower()

    for key, value in args.items():
        exp_type = expected_types.get(key, None)
        if exp_type is None:
            # If we do not have a type specification, leave the value as is.
            normalized_args[key] = value
        else:
            try:
                if exp_type.startswith("list"):
                    # Expected a list. If not a list, try to split or wrap it.
                    if not isinstance(value, list):
                        if isinstance(value, str):
                            # Split by comma, then strip whitespace.
                            normalized_args[key] = [item.strip() for item in value.split(",") if item.strip()]
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
                    # (Here we check if there's a dot in the string representation.)
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
                print(f"DEBUG: Error normalizing parameter '{key}' with value '{value}': {e}")
                normalized_args[key] = value
    return normalized_args

def main(
    model_ckpt: str = 'meta-llama/Meta-Llama-3-8B-Instruct',  # Updated to use LLaMA-2
    num_episodes: int = 15,
    num_stm_slots: int = 3,
    max_turn: int = 4,
    dir_write: str = "results/",
    if_visualize: bool = True,
):
    print("DEBUG: Entering main function with parameters:")
    print(f"DEBUG: model_ckpt={model_ckpt}, num_episodes={num_episodes}, num_stm_slots={num_stm_slots}, max_turn={max_turn}, dir_write={dir_write}, if_visualize={if_visualize}")

    # ----------------------------------------------
    print("DEBUG: Setting HF_HOME environment variable and creating cache directory if it does not exist.")
    HF_HOME = "/huggingface_cache"
    os.environ["HF_HOME"] = HF_HOME  # Ensure the environment variable is set
    # Create the cache directory if it doesn't exist
    os.makedirs(HF_HOME, exist_ok=True)

    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    print("DEBUG: Initializing tokenizer from MODEL_NAME.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME, max_length=4096)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    print("DEBUG: Initializing model from MODEL_NAME.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", cache_dir=HF_HOME
    )
    print(f"Model and tokenizer loaded successfully. Cached at {HF_HOME}")
    # ----------------------------------------------
    # Set the global tokenizer to avoid duplicate loading
    from my_llm import set_tokenizer
    set_tokenizer(tokenizer)
    
    print("DEBUG: Loading API descriptions and API list from JSON files.")
    with open("STE/tool_metadata/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)

    with open("STE/tool_metadata/API_list.json", "r", encoding='utf-8') as f:
        API_list = json.load(f)

    # (The unused hf_pipeline variable has been removed.)

    def run_evaluation(metric_name, args, truncate=2048):
        """
        Execute an evaluation metric from Hugging Face evaluate.
        """
        if metric_name not in API_list:
            raise ValueError(f"Metric '{metric_name}' is not supported. Supported metrics are: {API_list}")
        
        try:
            # Normalize the input arguments using the metadata.
            print(f"DEBUG: Normalizing evaluation arguments for metric '{metric_name}'")
            print(f"DEBUG: Arguments before normalization: {args}")
            normalized_args = normalize_evaluation_args(metric_name, args, API_descriptions)
            print("DEBUG: EVALUATIONEVALUATIONEVALUATIONEVALUATION: NORMALIZED ARGS = " + str(normalized_args))
            # Load the metric (this uses the default/cached model/tokenizer as defined in the metric's implementation)
            metric = evaluate.load(metric_name)
            # Compute the metric using the normalized arguments.
            result = metric.compute(**normalized_args)
            
            print("DEBUG: EVALUATIONEVALUATIONEVALUATIONEVALUATION RESULT = " + str(result))
            result_str = json.dumps(result)
            print("DEBUG: EVALUATIONEVALUATIONEVALUATIONEVALUATION RESULT JSONED = " + str(result_str))
            if truncate:
                result_str = result_str[:truncate]
            return result_str
        except Exception as e:
            print("DEBUG ERROR IN EVALUATIONEVALUATIONEVALUATIONEVALUATION: " + str(e))
            return f"Error in run_evaluation: {str(e)}"

    # Memory and reflection prompts (unchanged)
    PAST_Q_MSG_pre = "Below are queries you have already explored and whether you successfully solved them with the API's help:"
    PAST_Q_MSG_post = "Based on these, try to explore queries that can help you understand the API further; avoid synthesizing queries that are too close to the existing ones."
    prompt_reflection = "Do you think you successfully fulfilled this query in the end? Respond with \"Yes\" or \"No\"."

    print("DEBUG: Ensuring output directory exists.")
    os.makedirs(dir_write, exist_ok=True)
    data_dict = dict()

    print("DEBUG: Starting iteration over API_list.")
    for API in API_list:
        print("DEBUG: Processing API:", API)
        API_name_list = [API]

        with open("STE/prompts/prompt_explore.txt", "r") as f:
            prompt_template = f.read().strip()

        template_q, template_a, template_q_follow, template_a_follow = prompt_template.split("=========")
        template_q, template_a, template_q_follow, template_a_follow = template_q.strip(), template_a.strip(), template_q_follow.strip(), template_a_follow.strip()

        all_sessions, explored_queries, whether_successful = [], [], []

        for session_id in range(num_episodes):
            print("DEBUG: Starting episode:", session_id)
            item_list = []
            first_item = dict()
            api_descriptions_text = "\n\n".join(["API_name: {}\nDescription: {}".format(temp, API_descriptions[temp]) for temp in API_name_list])
            prompt_q = template_q.format(
                api_descriptions=api_descriptions_text,
            )

            if len(explored_queries) > 0:
                assert prompt_q.endswith("User Query:")
                prompt_q_added_question = strip_end(prompt_q, "User Query:").strip()
                prompt_q_added_question = prompt_q_added_question + "\n\n" + \
                    PAST_Q_MSG_pre + "\n" + "\n".join(LTM(explored_queries, whether_successful)) + \
                    "\n\n" + PAST_Q_MSG_post + "\n\nUser Query:"
            else:
                prompt_q_added_question = prompt_q

            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

            print("DEBUG: Generating the first query using chat_my.")
            response = chat_my(messages, prompt_q_added_question,
                               temp=1.0, stop="Thought:", visualize=if_visualize, max_tokens=360, model=model)[-1]['content']

            messages = messages + [
                {"role": "user", "content": prompt_q},
                {"role": "assistant", "content": response}
            ]

            query = messages[-1]['content']
            prompt_a = template_a.format(
                api_names=", ".join(API_name_list),
                query=query,
            )
            first_item['query'] = query
            explored_queries.append(query)

            chains = []
            print("DEBUG: Processing chain of calls for the first query.")
            messages = chat_my(messages, prompt_a, temp=1.0, stop="Observation:", visualize=if_visualize, max_tokens=360, model=model)
            temp = messages[-1]['content']
            parsed_response = parse_response(temp, API_name_list, API_descriptions)
            for _ in range(max_turn):
                if not parsed_response['parse_successful']:
                    observation = parsed_response['parse_error_msg']
                else:
                    if parsed_response['finish']:
                        chains.append(parsed_response)
                        break
                    else:
                        observation = run_evaluation(parsed_response['action'], parsed_response['action_input'])
                parsed_response['observation'] = observation
                chains.append(parsed_response)

                messages = chat_my(messages, "Observation: "+observation,
                                   temp=1.0, stop="Observation:", visualize=if_visualize, max_tokens=360, model=model)
                temp = messages[-1]['content']
                parsed_response = parse_response(temp, API_name_list, API_descriptions)

            if len(chains) == 0 or not chains[-1]['finish']:
                chains.append(parsed_response)

            first_item['chains'] = chains

            print("DEBUG: Running reflection to determine success for the first query.")
            messages = chat_my(messages, prompt_reflection, temp=1.0, stop="Observation:", visualize=if_visualize, max_tokens=360, model=model)
            res = messages[-1]['content']
            if "No" in res:
                successful = "No"
            else:
                successful = "Yes"

            whether_successful.append(successful)

            item_list.append(first_item)

            for _ in range(num_stm_slots-1):
                print("DEBUG: Processing a short-term memory slot.")
                item = dict()

                if len(explored_queries) > 0:
                    assert template_q_follow.endswith("User Query:")
                    template_q_follow_added_question = strip_end(template_q_follow, "User Query:").strip()
                    template_q_follow_added_question = template_q_follow_added_question + "\n\n" + \
                    PAST_Q_MSG_pre + "\n" + "\n".join(LTM(explored_queries, whether_successful)) + \
                    "\n\n" + PAST_Q_MSG_post + "\n\nUser Query:"
                else:
                    template_q_follow_added_question = template_q_follow

                print("DEBUG: Generating follow-up query using chat_my.")
                response = chat_my(messages, template_q_follow_added_question,
                                   temp=1.0, stop="Thought:", visualize=if_visualize, max_tokens=360, model=model)[-1]['content']
                messages = messages + [
                    {"role": "user", "content": template_q_follow},
                    {"role": "assistant", "content": response}
                ]

                query = messages[-1]['content']
                item['query'] = query
                explored_queries.append(query)

                chains = []
                print("DEBUG: Processing chain of calls for the short-term memory slot query.")
                messages = chat_my(messages, template_a_follow,
                                   temp=1.0, stop="Observation:", visualize=if_visualize, max_tokens=360, model=model)
                temp = messages[-1]['content']
                parsed_response = parse_response(temp, API_name_list, API_descriptions)
                for _ in range(max_turn):
                    if not parsed_response['parse_successful']:
                        observation = parsed_response['parse_error_msg']
                    else:
                        if parsed_response['finish']:
                            chains.append(parsed_response)
                            break
                        else:
                            observation = run_evaluation(parsed_response['action'], parsed_response['action_input'])
                    parsed_response['observation'] = observation
                    chains.append(parsed_response)

                    messages = chat_my(messages, "Observation: "+observation,
                                       temp=1.0, stop="Observation:", visualize=if_visualize, max_tokens=360, model=model)
                    temp = messages[-1]['content']
                    parsed_response = parse_response(temp, API_name_list, API_descriptions)

                if len(chains) == 0 or not chains[-1]['finish']:
                    chains.append(parsed_response)

                item['chains'] = chains

                print("DEBUG: Running reflection to determine success for the short-term memory slot query.")
                messages = chat_my(messages, prompt_reflection,
                                   temp=1.0, stop="Observation:", visualize=if_visualize, max_tokens=360, model=model)
                res = messages[-1]['content']
                if "No" in res:
                    successful = "No"
                else:
                    successful = "Yes"
                whether_successful.append(successful)

                item_list.append(item)

            all_sessions.append(
                {
                    "item_list": item_list,
                    "messages": messages,
                }
            )

            # Save partial / intermediate results right after each session
            print("DEBUG: Saving intermediate results for session.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            partial_filename = os.path.join(dir_write, f"intermediate_{API}_session_{session_id}_{timestamp}.json")
            try:
                with open(partial_filename, "w", encoding='utf-8') as f:
                    json.dump({
                        "API": API,
                        "session_id": session_id,
                        "all_sessions_so_far": all_sessions
                    }, f, ensure_ascii=False, indent=2)
                print(f"DEBUG: Intermediate results saved to {partial_filename}")
            except Exception as e:
                print(f"DEBUG: Error saving intermediate results: {e}")

        data_dict[API] = all_sessions

    print("DEBUG: Writing final data_dict to JSON file.")
    final_data_path = os.path.join(dir_write, f"data_dict_{timestamp}.json")
    with open(final_data_path, "w", encoding='utf-8') as f:
        json.dump(data_dict, f)
    print(f"DEBUG: Final data saved to {final_data_path}")
    print("DEBUG: Finished main function.")

if __name__ == '__main__':
    fire.Fire(main)