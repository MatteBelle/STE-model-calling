from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from termcolor import colored
from copy import deepcopy
import os
import requests  # For HTTP calls to the llm_server
import json

# Set up the Hugging Face cache directory
HF_HOME = "/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

def get_chat_completion_my(messages, max_tokens=512, temp=0.4, return_raw=False, stop=None):
    """
    Generate a response using the LLaMA model via the llm_server.
    This function builds the prompt following the ChatFormat guidelines for llama3-8B.
    """
    print("DENTRO CHAT COMPLETION MY", flush = True)
    server_url = os.environ.get("MODEL_SERVER_URL", "http://localhost:8000/generate")
    payload = {
        #"prompt": full_prompt,
        "prompt": messages,
        "max_tokens": max_tokens,
        "temperature": temp
    }
    is_response_empty = False
    try:
        response = requests.post(server_url, json=payload)
        response_json = response.json()
        response_text = response_json.get("response", "")
    except Exception as e:
        print("ERROR: Failed to get response from LLM server:", e)
        response_text = ""
    # Remove the prompt from the answer (cut from the head)
    parts = response_text.split("<|promptends|>")
    #print("PARTS: ", str(parts))
    if len(parts) > 1:
        response_text = parts[1].strip()
    else:
        print("DEBUG ERROR: response is empty OR <|promptends|> not found")
        response_text = "The response is empty."
        is_response_empty = True
    
    # Remove the prompt from the answer (cut from the head)
    if stop and stop in response_text:
        response_text = response_text.split(stop)[0].strip()
        
    print("ESCO CHAT COMPLETION MY", flush = True)
    # return response_text if not return_raw else {"response": response_text}
    return (response_text, is_response_empty)

def visualize_messages(messages):
    """
    Print messages in color for better readability.
    """
    role2color = {'system': 'red', 'assistant': 'green', 'user': 'cyan'}
    for entry in messages:
        assert entry['role'] in role2color.keys()
        if entry['content'].strip() != "":
            #print("GENERATED RESPONSE BEGINS: ----------------------")
            a=1
            #print(colored(entry['content'], role2color[entry['role']]))
            #print("GENERATED RESPONSE ENDS: ----------------------")
        else:
            #print("GENERATED RESPONSE BEGINS: ----------------------")
            print(colored("<no content>", role2color[entry['role']]))
            #print("GENERATED RESPONSE ENDS: ----------------------")

def chat_my(messages, new_message, visualize=True, **params):
    print("INSIDE CHAT MY", flush = True)
    # print("NEW MESSAGE: ", new_message)
    # print("NEW MESSAGE TYPE: ", type(new_message))
    messages = deepcopy(messages)
    messages.append({"role": "user", "content": new_message + "<|promptends|>"})
    print("INSIDE CHAT MY JSONED new_message is ", new_message, flush=True)
    #print("MESSAGE: ", str(messages[-1]['content']))
    # Call get_chat_completion_my without passing model
    response, is_response_empty = get_chat_completion_my(messages, **params)
    print("INSIDE CHAT MY response is ", response)
    messages[-1]["content"] = messages[-1]["content"].replace("<|promptends|>", "").strip()
    # clean the output of extra remaining of the chat template (llama-3-8b)
    response = response.replace('"assistant\n\n', "")
    response = response.replace('assistant\n', "")
    messages.append({"role": "assistant", "content": response})
    #print("DEBUG CHAT MY, RESPONSE: " + response)
    if visualize:
        visualize_messages(messages[-2:])
    print("ESCO DA CHAT MY", flush = True)
    return (messages, is_response_empty)