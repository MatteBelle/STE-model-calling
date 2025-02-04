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
    set_tokenizer(tokenizer)
    
    print("DEBUG: Loading API descriptions and API list from JSON files.")
    with open("STE/tool_metadata/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)

    with open("STE/tool_metadata/API_list.json", "r", encoding='utf-8') as f:
        API_list = json.load(f)

    # Initialize Hugging Face pipeline
    hf_pipeline = pipeline

    # def run_tool(full_API_name, args, truncate=2048):
    #     print("DEBUG: Entering run_tool function with full_API_name:", full_API_name)
    #     """
    #     Execute the Hugging Face pipeline API.
    #     """
    #     if full_API_name != "huggingface_pipeline":
    #         raise ValueError("Only the 'huggingface_pipeline' API is supported.")

    #     try:
    #         # Parse the input arguments
    #         print("DEBUG: RETRIEVING TASK")
    #         task = args.get("task")
    #         print("DEBUG: RETRIEVING MODEL")
    #         model = args.get("model", None)
    #         print("DEBUG: RETRIEVING INPUTS")
    #         inputs = args.get("inputs")
    #         print("DEBUG: RETRIEVING KWARGS")
    #         kwargs = args.get("kwargs", {})

    #         # Initialize the pipeline
    #         print("DEBUG: Initializing pipeline with task:", task, "model:", model)
    #         pipe = hf_pipeline(task, model=model_ckpt)

    #         # Execute the pipeline
    #         print("DEBUG: Executing pipeline call.")
    #         result = pipe(inputs, **kwargs)

    #         # Truncate the result if necessary
    #         if truncate:
    #             result = str(result)[:truncate]

    #         return json.dumps(result)
    #     except Exception as e:
    #         return f"Error in run_tool: {str(e)}"
    


    def run_evaluation(metric_name, args, truncate=2048):
        """
        Execute an evaluation metric from Hugging Face evaluate.
        """
        # Check that the metric is in our list (API_list)
        if metric_name not in API_list:
            raise ValueError(f"Metric '{metric_name}' is not supported. Supported metrics are: {API_list}")
        
        try:
            # Load the metric (this uses the default/cached model/tokenizer as defined in the metric's implementation)
            metric = evaluate.load(metric_name)
            # Compute the metric using the provided arguments.
            result = metric.compute(**args)
            
            # Optionally, truncate the stringified result
            result_str = json.dumps(result)
            if truncate:
                result_str = result_str[:truncate]
            return result_str
        except Exception as e:
            return f"Error in run_evaluation: {str(e)}"

    # Memory and reflection prompts
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
            api_descriptions = "\n\n".join(["API_name: {}\nDescription: {}".format(temp, API_descriptions[temp]) for temp in API_name_list])
            prompt_q = template_q.format(
                api_descriptions=api_descriptions,
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
            parsed_response = parse_response(temp, API_name_list, api_descriptions)
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
                parsed_response = parse_response(temp, API_name_list, api_descriptions)

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
                parsed_response = parse_response(temp, API_name_list, api_descriptions)
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
                    parsed_response = parse_response(temp, API_name_list, api_descriptions)

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
                    # Here we store the sessions accumulated so far for this API
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
    final_data_path = os.path.join(dir_write, "data_dict.json")
    with open(final_data_path, "w", encoding='utf-8') as f:
        json.dump(data_dict, f)
    print(f"DEBUG: Final data saved to {final_data_path}")
    print("DEBUG: Finished main function.")

if __name__ == '__main__':
    fire.Fire(main)