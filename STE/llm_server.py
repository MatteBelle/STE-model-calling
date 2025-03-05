from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import List, Dict

# Set up the Hugging Face cache directory
HF_HOME = "/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

#MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
print("DEBUG (Server): Initializing tokenizer from MODEL_NAME.", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME, max_length=4096)
print(f"DEBUG (Server): Model and tokenizer loaded successfully. Cached at {HF_HOME}", flush=True)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})

print("DEBUG (Server): Initializing model from MODEL_NAME.", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
    cache_dir=HF_HOME
)

app = FastAPI()

# Define the request schema
class InferenceRequest(BaseModel):
    prompt: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.4

def dynamic_context_trim(messages, max_length=8192, reserve_tokens=768):
    """
    Dynamically trims conversation history to fit within context window.
    
    Strategy:
    1. Always keep system messages
    2. Always keep the last user message (current query)
    3. Remove older messages as complete blocks (user+assistant pairs)
    4. Reserve space for the model response
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_length: Maximum token length allowed
        reserve_tokens: Number of tokens to reserve for generation
        
    Returns:
        Tuple of (trimmed_messages, was_trimmed, token_count)
    """
    # Ensure we have something to work with
    if not messages:
        return messages, False, 0
    
    # Apply chat template to full context to check length
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(full_prompt, return_tensors="pt")
    token_count = len(tokens.input_ids[0])
    
    # If we're under the limit considering reserved tokens, no trimming needed
    effective_max_length = max_length - reserve_tokens
    if token_count <= effective_max_length:
        return messages, False, token_count
    
    # Always keep system messages
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    non_system_messages = [msg for msg in messages if msg["role"] != "system"]
    
    # Always keep the last user message (the current query)
    last_user_index = -1
    for i in range(len(non_system_messages) - 1, -1, -1):
        if non_system_messages[i]["role"] == "user":
            last_user_index = i
            break
    
    if last_user_index >= 0:
        current_query = non_system_messages[last_user_index:]
        history = non_system_messages[:last_user_index]
    else:
        current_query = []
        history = non_system_messages
    
    # Progressive removal of older message pairs until we fit the context
    trimmed_messages = system_messages + history + current_query
    was_trimmed = False
    
    while history and len(tokenizer.apply_chat_template(trimmed_messages, tokenize=False)) > effective_max_length:
        # Remove the oldest pair (or single message if odd number)
        history = history[2:] if len(history) >= 2 else history[1:] if history else []
        trimmed_messages = system_messages + history + current_query
        was_trimmed = True
    
    # Final check of token count
    final_prompt = tokenizer.apply_chat_template(trimmed_messages, tokenize=False)
    final_token_count = len(tokenizer(final_prompt, return_tensors="pt").input_ids[0])
    
    # Log trimming information
    if was_trimmed:
        removed_count = len(messages) - len(trimmed_messages)
        print(f"DEBUG (Server): Context trimmed - removed {removed_count} messages. " 
              f"Tokens: {token_count} → {final_token_count}", flush=True)
    
    return trimmed_messages, was_trimmed, final_token_count

# Add this route to llm_server.py
@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is running"""
    return {"status": "ok", "port": int(os.environ.get("SERVER_PORT", 8000))}

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    try:
        # Apply dynamic context trimming
        trimmed_prompt, was_trimmed, token_count = dynamic_context_trim(
            request.prompt, 
            max_length=8192,
            reserve_tokens=request.max_tokens
        )
        
        # Tokenize the trimmed prompt
        inputs = tokenizer(
            tokenizer.apply_chat_template(trimmed_prompt, tokenize=False),
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to("cuda")
        
        print(f"DEBUG (Server): Prompt length (tokens): {token_count}", flush=True)
        
        # Generate output using the provided parameters
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
            )
        
        # Decode the full generated output
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        del inputs, output_tokens
        torch.cuda.empty_cache()
        
        # Return the response with trimming metadata
        return {
            "response": generated_text,
            "context_trimmed": was_trimmed,
            "token_count": token_count
        }
    except Exception as e:
        # Ensure memory is cleared even on error
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SERVER_PORT", 8000))
    print(f"Starting server on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)