import json
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import requests
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from utils import find_reverse, random_choose, parse_response, strip_end

# ----------------------------------------------
# Set a persistent local cache directory for Hugging Face
HF_HOME = "/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME  # Ensure the environment variable is set
# Create the cache directory if it doesn't exist
os.makedirs(HF_HOME, exist_ok=True)
# Load the LLaMA 2 model and tokenizer, using the cache
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME)
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", cache_dir=HF_HOME
)
print(f"Model and tokenizer loaded successfully. Cached at {HF_HOME}")
# ----------------------------------------------

def LTM(X, labels):
    assert len(X) == len(labels)
    return ["Query: {} | Solved: {}".format(X[i], labels[i]) for i in range(len(X))]

def main(
    model_ckpt: str = 'llama-2-7b-hf',
    num_episodes: int = 15,
    num_stm_slots: int = 3,
    max_turn: int = 4,
    dir_write: str = "results/",
    rapidapi_key: str = "",
    if_visualize: bool = True,
):

    # with open("BMTools/secret_keys.sh", "r", encoding='utf-8') as f:
    #     env_variables = f.readlines()
    # for var in env_variables:
    #     if var.strip() == "":
    #         continue
    #     key, val = var.replace("export", "").strip().split("=")
    #     val = val.strip("'")
    #     os.environ[key] = val

    with open("STE/tool_metadata/tool2cate.json", "r", encoding='utf-8') as f:
        tool2cate = json.load(f)

    with open("STE/tool_metadata/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)

    with open("STE/tool_metadata/API_list.json", "r", encoding='utf-8') as f:
        API_list = json.load(f)

    # with open("STE/tool_metadata/API_dict_bmtools.json", "r", encoding='utf-8') as f:
    #     API_dict_bmtools = json.load(f)

    bm_tools_l = ['search',
                 'disambiguation',
                 'search_places',
                 'getWolframAlphaResults',
                 'get_weather_today',
                 'forecast_weather',
                 'get_arxiv_article_information',
                 'get_today_date',
                 'add_date',
                 'search_general']

    name_to_tool_map_global = dict()

    # Remove bmtools code as instructed
    # For now, we remove loading tools and run_tool function for bmtools

    def run_tool(full_API_name, args, truncate=2048):
        pass  # Placeholder, since bmtools and toolbench are not used

    PAST_Q_MSG_pre = "Below are queries you have already explored and whether you successfully solved them with the API's help:"
    PAST_Q_MSG_post = "Based on these, try to explore queries that can help you understand the API further; avoid synthesizing queries that are too close to the existing ones."

    prompt_reflection = "Do you think you successfully fulfilled this query in the end? Respond with \"Yes\" or \"No\"."

    os.makedirs(dir_write, exist_ok=True)
    data_dict = dict()

    for API in API_list:

        print("===== Currently", API, "=====")

        API_name_list = [API]

        with open("/STE/prompts/prompt_explore.txt", "r") as f:
            prompt_template = f.read().strip()

        template_q, template_a, template_q_follow, template_a_follow = prompt_template.split("=========")
        template_q, template_a, template_q_follow, template_a_follow = template_q.strip(), template_a.strip(), template_q_follow.strip(), template_a_follow.strip()

        all_sessions, explored_queries, whether_successful = [], [], []

        for session_id in range(num_episodes):
            print("==== episode ====", session_id)
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

            # Using LLaMA instead of GPT
            inputs = tokenizer(prompt_q_added_question, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=512, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
            inputs = tokenizer(prompt_a, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=360, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
            temp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed_response = parse_response(temp, API_name_list, api_descriptions)
            for _ in range(max_turn):
                if not parsed_response['parse_successful']:
                    observation = parsed_response['parse_error_msg']
                else:
                    if parsed_response['finish']:
                        chains.append(parsed_response)
                        break
                    else:
                        observation = run_tool(parsed_response['action'], parsed_response['action_input'])
                parsed_response['observation'] = observation
                chains.append(parsed_response)

                inputs = tokenizer("Observation: "+observation, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_length=360, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
                temp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                parsed_response = parse_response(temp, API_name_list, api_descriptions)

            if len(chains) == 0 or not chains[-1]['finish']:
                chains.append(parsed_response)

            first_item['chains'] = chains

            inputs = tokenizer(prompt_reflection, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=360, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "No" in res:
                successful = "No"
            else:
                successful = "Yes"

            whether_successful.append(successful)

            item_list.append(first_item)

            for _ in range(num_stm_slots-1):
                print("----------------------------------------")
                item = dict()

                if len(explored_queries) > 0:
                    assert template_q_follow.endswith("User Query:")
                    template_q_follow_added_question = strip_end(template_q_follow, "User Query:").strip()
                    template_q_follow_added_question = template_q_follow_added_question + "\n\n" + \
                    PAST_Q_MSG_pre + "\n" + "\n".join(LTM(explored_queries, whether_successful)) + \
                    "\n\n" + PAST_Q_MSG_post + "\n\nUser Query:"
                else:
                    template_q_follow_added_question = template_q_follow

                inputs = tokenizer(template_q_follow_added_question, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_length=360, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                messages = messages + [
                    {"role": "user", "content": template_q_follow},
                    {"role": "assistant", "content": response}
                ]

                query = messages[-1]['content']
                item['query'] = query
                explored_queries.append(query)

                chains = []
                inputs = tokenizer(template_a_follow, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_length=360, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
                temp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                parsed_response = parse_response(temp, API_name_list, api_descriptions)
                for _ in range(max_turn):
                    if not parsed_response['parse_successful']:
                        observation = parsed_response['parse_error_msg']
                    else:
                        if parsed_response['finish']:
                            chains.append(parsed_response)
                            break
                        else:
                            observation = run_tool(parsed_response['action'], parsed_response['action_input'])
                    parsed_response['observation'] = observation
                    chains.append(parsed_response)

                    inputs = tokenizer("Observation: "+observation, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_length=360, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
                    temp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    parsed_response = parse_response(temp, API_name_list, api_descriptions)

                if len(chains) == 0 or not chains[-1]['finish']:
                    chains.append(parsed_response)

                item['chains'] = chains

                inputs = tokenizer(prompt_reflection, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_length=360, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
                res = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

        data_dict[API] = all_sessions

    with open(os.path.join(dir_write, "data_dict.json"), "w", encoding='utf-8') as f:
        json.dump(data_dict, f)


if __name__ == '__main__':
    fire.Fire(main)