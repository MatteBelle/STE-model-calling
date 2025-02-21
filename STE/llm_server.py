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

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
print("DEBUGs (Server): Initializing tokenizer from MODEL_NAME.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME, max_length=4096)
# test = [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#   {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]
# print("CHAT_TEMPLATE APPLIED TO TEST: " + str(tokenizer.apply_chat_template(test, tokenize=False)))
#model.resize_token_embeddings(len(tokenizer))
print(f"DEBUG (Server): Model and tokenizer loaded successfully. Cached at {HF_HOME}")

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
print("DEBUG (Server): Initializing model from MODEL_NAME.")
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
        print("DENTRO GENERATEEEE------------------------------------------------------------------------------")
        print("REQUEST PROMPT BEFORE TOKENIZATION: ", request.prompt)
        print("REQUEST PROMPT AFTER TOKENIZATION: ", tokenizer.apply_chat_template(request.prompt, tokenize=False))
        inputs = tokenizer(tokenizer.apply_chat_template(request.prompt, tokenize=False), return_tensors="pt").to("cuda")
        token_ids = inputs.input_ids[0].tolist()
        print("DEBUG (Server): Prompt length (tokens):", len(token_ids))
        print("GENERANDO RISPOSTA GENERATEEEE------------------------------------------------------------------------------")
        # Generate output using the provided parameters
        output_tokens = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            #pad_token_id=tokenizer.pad_token_id,
        )
        # Decode the full generated output
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        if generated_text.startswith("assistant\n"):
            generated_text = generated_text[len("assistant\n"):].strip()
        return {"response": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)