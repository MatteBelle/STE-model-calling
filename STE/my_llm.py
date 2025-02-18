from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from termcolor import colored
from copy import deepcopy
import os
import requests  # For HTTP calls to the llm_server

# Set up the Hugging Face cache directory
HF_HOME = "/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

def get_chat_completion_my(messages, max_tokens=512, temp=0.1, return_raw=False, stop=None):
    # Build the prompt as before.
    prompt = format_messages(messages)
    
    # Append a unique delimiter to clearly separate prompt from generation.
    delimiter = "\n<|startofresponse|>\n"
    full_prompt = prompt + delimiter
    print("DEBUG: Formatted prompt with delimiter:", full_prompt)
    
    # Prepare payload using the full prompt (with delimiter).
    server_url = os.environ.get("MODEL_SERVER_URL", "http://localhost:8000/generate")
    payload = {
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temp
    }
    print("DEBUG: Sending payload to LLM server:", payload)
    
    try:
        response = requests.post(server_url, json=payload)
        response_json = response.json()
        response_text = response_json.get("response", "")
    except Exception as e:
        print("ERROR: Failed to get response from LLM server:", e)
        response_text = ""
    
    print(f"DEBUG-MY_LLM: Full response before splitting:\n{response_text}")
    print(f"DEBUG-MY_LLM-LINE55: Length of prompt: {len(prompt)}")
    print(f"DEBUG-MY_LLM-LINE55: Length of response: {len(response_text)}")
    
    # Now remove everything up to (and including) the delimiter.
    if delimiter in response_text:
        response_text = response_text.split(delimiter, 1)[1].strip()
    else:
        # Fallback if the delimiter is not found.
        response_text = response_text[len(full_prompt):].strip()
    
    print(f"DEBUG-MY_LLM: Full response after splitting:\n{response_text}")
    
    # Apply stop condition if provided.
    if stop and stop in response_text:
        response_text = response_text.split(stop)[0].strip()
    
    return response_text if not return_raw else {"response": response_text}

def format_messages(messages):
    """
    Format messages in a chat-friendly manner.
    """
    formatted = ""
    for msg in messages:
        print("DEBUG DEBUG DEBUG - ROLES: " + msg["role"])
        if msg["role"] == "system":
            formatted += f"[SYSTEM]: {msg['content']}\n"
        elif msg["role"] == "user":
            formatted += f"[USER]: {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"[ASSISTANT]: {msg['content']}\n"
    return formatted

def visualize_messages(messages):
    """
    Print messages in color for better readability.
    """
    role2color = {'system': 'red', 'assistant': 'green', 'user': 'cyan'}
    for entry in messages:
        assert entry['role'] in role2color.keys()
        if entry['content'].strip() != "":
            print("GENERATED RESPONSE BEGINS: ----------------------")
            print(colored(entry['content'], role2color[entry['role']]))
            print("GENERATED RESPONSE ENDS: ----------------------")
        else:
            print("GENERATED RESPONSE BEGINS: ----------------------")
            print(colored("<no content>", role2color[entry['role']]))
            print("GENERATED RESPONSE ENDS: ----------------------")

def chat_my(messages, new_message, visualize=True, **params):
    messages = deepcopy(messages)
    messages.append({"role": "user", "content": new_message})
    # Call get_chat_completion_my without passing model
    response = get_chat_completion_my(messages, **params)
    messages.append({"role": "assistant", "content": response})
    print("DEBUG CHAT MY, RESPONSE: " + response)
    if visualize:
        visualize_messages(messages[-2:])
    return messages
