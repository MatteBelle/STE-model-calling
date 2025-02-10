from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from termcolor import colored
from copy import deepcopy
import os
import requests  # New import for HTTP calls

# Set up the Hugging Face cache directory
HF_HOME = "/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

tokenizer = None

def set_tokenizer(new_tokenizer):
    """
    Set the global tokenizer to avoid duplicate loading.
    """
    global tokenizer
    tokenizer = new_tokenizer

# URL of the inference server (adjust if necessary)
LLM_SERVER_URL = "http://localhost:8000/generate"

def get_chat_completion_my(messages, model=None, max_tokens=512, temp=0.7, return_raw=False, stop=None):
    """
    Generate a response using the LLM server.
    """
    if model is None:
        raise ValueError("A valid model instance must be provided to get_chat_completion_my")
    if tokenizer is None:
        raise ValueError("Tokenizer has not been set. Please call set_tokenizer with a valid tokenizer.")

    # Format messages (all the chat formatting logic stays here)
    prompt = format_messages(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_ids = inputs.input_ids[0].tolist()
    print("DEBUG: Prompt length (tokens):", len(token_ids))
    
    # Instead of performing local generation, send an HTTP request to the LLM server
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temp
    }
    print("DEBUG: Sending request to LLM server with payload:")
    print(payload)
    try:
        response = requests.post(LLM_SERVER_URL, json=payload)
        response_json = response.json()
        response_text = response_json.get("response", "")
    except Exception as e:
        print("ERROR: Failed to get response from LLM server:", e)
        response_text = ""
    
    # Debug prints as before
    print(f"DEBUG-MY_LLM-LINE55: Full response before slicing:\n{response_text}")
    print(f"DEBUG-MY_LLM-LINE55: Length of prompt: {len(prompt)}")
    print(f"DEBUG-MY_LLM-LINE55: First {len(prompt)} characters of response: {response_text[:len(prompt)]}")

    # Remove the prompt from the response
    response_text = response_text[len(prompt):].strip()
    print(f"DEBUG-MY_LLM-LINE55: Full response after slicing:\n{response_text}")
    
    # Apply stop condition if provided
    if stop and stop in response_text:
        response_text = response_text.split(stop)[0].strip()

    return response_text if not return_raw else {"response": response_text}

def format_messages(messages):
    """
    Format messages in a chat-friendly manner.
    A unique marker (<|endofprompt|>) is appended to the end to help extract the generated text.
    """
    formatted = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted += f"[SYSTEM]: {msg['content']}\n"
        elif msg["role"] == "user":
            formatted += f"[USER]: {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"[ASSISTANT]: {msg['content']}\n"
    #formatted += "<|endofprompt|>" # NEW MODIFICATION (left commented as before)
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
    response = get_chat_completion_my(messages, **params)
    messages.append({"role": "assistant", "content": response})
    print("DEBUG CHAT MY, RESPONSE: " + response)
    if visualize:
        filtered_messages = [msg for msg in messages if msg['content'].strip()]
        visualize_messages(filtered_messages[-2:])
