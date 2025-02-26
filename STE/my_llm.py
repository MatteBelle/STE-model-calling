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
    # Build the formatted prompt using the new ChatFormat
    # prompt = format_messages(messages)
    
    # # Append the assistant header to signal that the assistant should now respond.
    # assistant_header = "<|start_header_id|> assistant <|end_header_id|>\n\n"
    # full_prompt = prompt + assistant_header
    # print("DEBUG: Formatted prompt with assistant header:", full_prompt)
    # print("MESSAGES: ", str(messages))
    # print("MESSAGES TYPE: ", type(messages))
    print("DENTRO CHAT COMPLETION MY", flush = True)
    server_url = os.environ.get("MODEL_SERVER_URL", "http://localhost:8000/generate")
    payload = {
        #"prompt": full_prompt,
        "prompt": messages,
        "max_tokens": max_tokens,
        "temperature": temp
    }

    try:
        # print("MESSAGES TYPESSS: ", type(messages))
        # print("MESSAGES max_tokens TYPESSS: ", type(max_tokens))
        # print("MESSAGES temperature TYPESSS: ", type(temp))
        # for d in messages:
        #     print(type(d))
        #     print(type(d['role']))
        #     print(type(d['content']))
        response = requests.post(server_url, json=payload)
        response_json = response.json()
        response_text = response_json.get("response", "")
    except Exception as e:
        print("ERROR: Failed to get response from LLM server:", e)
        response_text = ""
    #print("RESPONSE TEXT: ", response_text)
    #print(f"DEBUG-MY_LLM: Full response before processing:\n{response_text}")
    
    # Remove the prompt part from the generated text.
    # if response_text.startswith(full_prompt):
    #     response_text = response_text[len(full_prompt):].strip()
    # else:
    #     # Fallback: if the full prompt is not found, split by the eot token.
    #     parts = response_text.split("<|eot_id|>") #FAKE: IN SERVER I ALREADY USE skip_special_tokens=True
    #     if parts:
    #         response_text = parts[0].strip()
    # Fallback: if the full prompt is not found, split by the eot token.
    #parts = response_text.split("<|eot_id|>")
    
    # Remove the prompt from the answer (cut from the head)
    parts = response_text.split("<|promptends|>")
    #print("PARTS: ", str(parts))
    if parts:
        response_text = parts[1].strip()
    else:
        print("DEBUG ERROR: <|promptends|> not found")
    
    # Remove the prompt from the answer (cut from the head)
    if stop and stop in response_text:
        response_text = response_text.split(stop)[0].strip()
    
    # Remove any stray tokens (for instance, <|endofresponse|>) ----> FAKE: IN SERVER I ALREADY USE skip_special_tokens=True
    #response_text = response_text.replace("<|endofresponse|>", "").strip()
    print("ESCO CHAT COMPLETION MY", flush = True)
    return response_text if not return_raw else {"response": response_text}

def format_messages(messages):
    """
    Format messages according to Llama3-8B ChatFormat.
    The format is:
      <|begin_of_text|>
      <|start_header_id|> role <|end_header_id|>
      
      message content <|eot_id|>
      (repeat for each message)
    """
    # formatted = "<|begin_of_text|>\n"
    # for msg in messages:
    #     role = msg["role"].lower()  # ensure the role is in lower case if needed
    #     # Add header for the message
    #     formatted += f"<|start_header_id|> {role} <|end_header_id|>\n\n"
    #     # Add the message content and mark end of message
    #     formatted += f"{msg['content']} <|eot_id|>\n"
    return formatted

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
    print("DENTRO CHAT MY", flush = True)
    # print("NEW MESSAGE: ", new_message)
    # print("NEW MESSAGE TYPE: ", type(new_message))
    messages = deepcopy(messages)
    messages.append({"role": "user", "content": new_message + "<|promptends|>"})
    print("DENTRO CHAT MY JSONED new_message is ", new_message + "<|promptends|>", flush=True)
    #print("MESSAGE: ", str(messages[-1]['content']))
    # Call get_chat_completion_my without passing model
    response = get_chat_completion_my(messages, **params)
    print("DENTRO CHAT MY response is ", response)
    messages[-1]["content"] = messages[-1]["content"].replace("<|promptends|>", "").strip()
    # clean the output of extra remaining of the chat template (llama-3-8b)
    response = response.replace('"assistant\n\n', "")
    response = response.replace('assistant\n', "")
    messages.append({"role": "assistant", "content": response})
    #print("DEBUG CHAT MY, RESPONSE: " + response)
    if visualize:
        visualize_messages(messages[-2:])
    print("ESCO DA CHAT MY", flush = True)
    return messages