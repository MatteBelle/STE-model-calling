from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from termcolor import colored
from copy import deepcopy
import os

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

def get_chat_completion_my(messages, model=None, max_tokens=512, temp=0.7, return_raw=False, stop=None):
    """
    Generate a response using LLaMA 2 model.
    """
    if model is None:
        raise ValueError("A valid model instance must be provided to get_chat_completion_my")
    if tokenizer is None:
        raise ValueError("Tokenizer has not been set. Please call set_tokenizer with a valid tokenizer.")

    # Format messages for LLaMA-2
    prompt = format_messages(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_ids = inputs.input_ids[0].tolist()
    print("DEBUG: Prompt length (tokens):", len(token_ids))
    
    # TODO TEST THIS CODE 
    max_context_length = tokenizer.model_max_length  # or your model's limit
    if inputs.input_ids.shape[1] > max_context_length:
        # Remove older messages (or summarize them)
        # Example: keep only the last N tokens
        inputs.input_ids = inputs.input_ids[:, -max_context_length:]
    # TODO TEST THIS CODE 
    
    # Generate response
    output_tokens = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + max_tokens,
        temperature=temp,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the response
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()  # Remove the prompt from the response

    # Apply stop condition
    if stop and stop in response:
        response = response.split(stop)[0].strip()

    return response if not return_raw else {"response": response}

def format_messages(messages):
    """
    Format messages in a chat-friendly way for LLaMA 2.
    """
    formatted = ""
    for msg in messages:
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
    """
    Chat with the LLaMA 2 model.
    """
    messages = deepcopy(messages)
    messages.append({"role": "user", "content": new_message})
    response = get_chat_completion_my(messages, **params)
    messages.append({"role": "assistant", "content": response})
    if visualize:
        visualize_messages(messages[-2:])
    return messages