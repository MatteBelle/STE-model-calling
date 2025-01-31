from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from termcolor import colored
from copy import deepcopy

# Set up the Hugging Face cache directory
HF_HOME = "/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# Load the LLaMA 2 model and tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME)
tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", cache_dir=HF_HOME
)
print(f"Model and tokenizer loaded successfully. Cached at {HF_HOME}")

def get_chat_completion_my(messages, max_tokens=512, temp=0.7, return_raw=False, stop=None):
    """
    Generate a response using LLaMA 2 model.
    """
    # Format messages for LLaMA-2
    prompt = format_messages(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
            print(colored(entry['content'], role2color[entry['role']]))
        else:
            print(colored("<no content>", role2color[entry['role']]))

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