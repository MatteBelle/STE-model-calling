#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import argparse
import logging
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetuning.log")
    ]
)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description="Fine-tune a Llama 3.1-8B model for metric evaluation")
parser.add_argument('--data-file', type=str, default="dataset.json", help="Path to the data file")
parser.add_argument('--output-dir', type=str, default="outputs/metric_evaluation_assistant", help="Output directory")
parser.add_argument('--batch-size', type=int, default=4, help="Batch size per GPU")
parser.add_argument('--grad-accum', type=int, default=4, help="Gradient accumulation steps")
parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
parser.add_argument('--max-seq-length', type=int, default=2048, help="Maximum sequence length")
args = parser.parse_args()

# Set up the Hugging Face cache directory
HF_HOME = os.environ.get("HF_HOME", "/huggingface_cache")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)
logger.info(f"Using Hugging Face cache directory: {HF_HOME}")

# Load and prepare the dataset
logger.info(f"Loading data from {args.data_file}")
try:
    with open(args.data_file, "r") as f:
        data_source = json.load(f)
except Exception as e:
    logger.error(f"Failed to load data file: {e}")
    raise

# Extract training examples from the data
training_examples = []

# Process the dataset.json entries
for entry in data_source:
    if "answer" in entry:
        # Create a conversation with system, user and assistant messages
        messages = []
        
        # Add system message if present
        if "system" in entry and entry["system"]:
            messages.append({
                "role": "system",
                "content": entry["system"]
            })
        
        # Add user message if present
        if "query" in entry:
            messages.append({
                "role": "user",
                "content": entry["query"]
            })
        
        # Add assistant message (answer)
        if entry["answer"]:
            messages.append({
                "role": "assistant",
                "content": entry["answer"]
            })
        
        # Only add examples with at least one valid message
        if messages:
            training_examples.append({
                "messages": messages
            })

logger.info(f"Extracted {len(training_examples)} training examples")

# Load the model
logger.info("Loading the model...")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Check for HF_TOKEN in environment or file
hf_token = os.environ.get("HF_TOKEN", None)
if not hf_token and os.path.exists("hf_token.txt"):
    with open("hf_token.txt", "r") as f:
        hf_token = f.read().strip()

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        cache_dir=HF_HOME,
        token=hf_token,
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Convert the data to a Hugging Face dataset
dataset = Dataset.from_list(training_examples)

# Define the formatting function for the Llama 3.1 chat format
def formatting_chat_func(examples):
    formatted_texts = []
    
    for example in examples["messages"]:
        # Filter out empty messages
        valid_messages = [msg for msg in example if isinstance(msg, dict) and "content" in msg and "role" in msg and msg["content"]]
        
        if len(valid_messages) < 1:  # Skip if no valid messages
            continue
        
        # Apply Llama 3.1 chat template directly
        formatted_text = tokenizer.apply_chat_template(valid_messages, tokenize=False)
        formatted_texts.append(formatted_text)
    
    return {"text": formatted_texts}

# Format the dataset
logger.info("Formatting dataset...")
formatted_dataset = dataset.map(
    formatting_chat_func,
    batched=True,
    remove_columns=dataset.column_names
)

# Filter out any empty examples
formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"]) > 0)
logger.info(f"Final dataset contains {len(formatted_dataset)} examples")

# Add LoRA adapters
logger.info("Adding LoRA adapters to the model...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Create the trainer
logger.info("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        report_to="tensorboard",
    ),
)

# Train the model
logger.info("Starting training...")
trainer.train()

# Save the model
save_dir = os.path.join(args.output_dir, "final_model")
logger.info(f"Training complete. Saving model to {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
logger.info("Model saved successfully")