from tqdm.notebook import tqdm
import pandas as pd
import os
import csv
import sys
import numpy as np
import time
import random
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import textwrap
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from adapters import AdapterConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using Device: {device}')

def print_trainable_params(model, stage_name="Model"):
    print(f"\nTrainable Parameters in {stage_name}:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.numel()} params")

# Load Llama 1B and tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Using LLama 1B as base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Use EOS token as PAD token
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
base_model.config.pad_token_id = base_model.config.eos_token_id


ds = load_dataset("stanfordnlp/imdb")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

# Tokenize datasets
tokenized_datasets = ds.map(preprocess_function, batched=True)

# Prepare train and test datasets
train_dataset = tokenized_datasets["train"].shuffle(seed=42)  # Use full training dataset
test_dataset = tokenized_datasets["test"].shuffle(seed=42)    # Use full testing dataset

# Veyr big dataset
# Load a sentiment dataset (example: SST2)
# ds = load_dataset("facebook/xnli", "all_languages")
# train_data = ds['train']
# val_data = ds['validation']
            
            
# Prepare training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate periodically during training
    #eval_steps=100,               # Frequency of evaluation (adjust as needed)
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training for GPU
    report_to="none",  # Disable reporting to avoid unnecessary overhead
)

# Train base model
trainer_base = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

base_model.to(device)

print_trainable_params(base_model, stage_name="Base Model")

print("\nTraining Base Model...")
# Resize model embeddings after adding new special tokens
base_model.resize_token_embeddings(len(tokenizer))
trainer_base.train()

# Save base model
tokenizer.save_pretrained("./base_model")
base_model.save_pretrained("./base_model")



# Evaluate base model
print("\nEvaluating Base Model...")
base_results = trainer_base.evaluate()
print("Base Model Results:", base_results)
