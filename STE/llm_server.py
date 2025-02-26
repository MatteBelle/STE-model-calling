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

# device = torch.device("cuda:0")
# # Set PyTorch to use only 90% of the GPU memory
# torch.cuda.set_per_process_memory_fraction(0.9, device=device)

#MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
print("DEBUGs (Server): Initializing tokenizer from MODEL_NAME.", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME, max_length=4096)
print(f"DEBUG (Server): Model and tokenizer loaded successfully. Cached at {HF_HOME}", flush=True)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
# test = [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#   {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]
# print("CHAT_TEMPLATE APPLIED TO TEST: " + str(tokenizer.apply_chat_template(test, tokenize=False)))
print("DEBUG (Server): Initializing model from MODEL_NAME.", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    #device_map="auto",
    device_map="cuda",
    cache_dir=HF_HOME
)

app = FastAPI()

# Define the request schema; note that all the logic (formatting, slicing) is done on the client.
class InferenceRequest(BaseModel):
    prompt: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.4

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    try:
        # Tokenize the provided prompt (which is already formatted by the client)
        #inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        print("DENTRO GENERATEEEE------------------------------------------------------------------------------", flush=True)
        print("REQUEST PROMPT BEFORE TOKENIZATION: ", request.prompt, flush=True)
        print("REQUEST PROMPT AFTER TOKENIZATION: ", tokenizer.apply_chat_template(request.prompt, tokenize=False), flush=True)
        inputs = tokenizer(
            tokenizer.apply_chat_template(request.prompt, tokenize=False),
            return_tensors="pt",
            truncation=True,
            max_length=20000
            ).to("cuda")
        token_ids = inputs.input_ids[0].tolist()
        print("DEBUG (Server): Prompt length (tokens):", len(token_ids), flush=True)
        print("GENERANDO RISPOSTA GENERATEEEE------------------------------------------------------------------------------", flush=True)
        # Generate output using the provided parameters
        #with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            #pad_token_id=tokenizer.pad_token_id,
        )
        # Decode the full generated output
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        del inputs, output_tokens
        torch.cuda.empty_cache()
        print("GENERATED TEXT= ", str(generated_text), flush=True)
        return {"response": generated_text}
    except Exception as e:
        # Ensure memory is cleared even on error
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)