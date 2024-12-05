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

from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path

from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaEncoder,
    RobertaLayer,
    RobertaEmbeddings,
    RobertaConfig,
)

from custom_Roberta import CustomRobertaModel, initialize_weights


# Define model directories
BASE_MODEL_DIR = "./base_model"
LORA_MODEL_DIR = "./lora_model"
ADAPTER_MODEL_DIR = "./adapter_model"

# Base directory names
from datetime import datetime

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_results_dir = f"./results/base" #_{timestamp}"
lora_results_dir = f"./results/lora" #_{timestamp}"
adapter_results_dir = f"./results/adapter" #_{timestamp}"

# Ensure directories exist
Path(base_results_dir).mkdir(parents=True, exist_ok=True)
Path(lora_results_dir).mkdir(parents=True, exist_ok=True)
Path(adapter_results_dir).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using Device: {device}')

#Number of epochs for all models
num_epochs = 5

def print_trainable_params(model, stage_name="Model"):
    print(f"\nTrainable Parameters in {stage_name}:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"%\\age of trainable params: {(trainable_params/total_params) * 100}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.numel()} params")

# Base TrainingArguments configuration
base_args = {
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": num_epochs,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "load_best_model_at_end": True,
    "fp16": True,
    "report_to": "none",
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}
# To create dynamic result directory
def create_training_args(output_dir, lr):
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        learning_rate = lr,
        **base_args,
    )

# Load Llama 1B and tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Using LLama 1B as base model

# Couldn't train Llama because of lower mem GPUs so shifting to roberta
model_name = "FacebookAI/roberta-large"

# Step 1: Load or initialize tokenizer
if os.path.exists(BASE_MODEL_DIR):
    print("\nTokenizer already exists. Loading from base model directory...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
else:
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD token


# Loading Dataset
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
            

# Function to evaluate models
def evaluate_model(model, training_args, name):
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    print(f"\nEvaluating {name} Model...")
    results = trainer.evaluate()
    print(f"{name} Model Results:", results)
    return results



# Step 2: Train or load the base model
base_training_args = create_training_args(output_dir=base_results_dir, lr=2e-5)

if os.path.exists(BASE_MODEL_DIR):
    print("\nBase model already exists. Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_DIR)
else:
    print("\nTraining Base Model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.pad_token_id = base_model.config.eos_token_id
    trainer_base = Trainer(
        model=base_model,
        args=base_training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print_trainable_params(base_model, stage_name="Base Model")
    base_model.to(device)
    start_time = time.time()
    trainer_base.train()
    print(f"Base Model training time: {time.time() - start_time}s")
    base_model.save_pretrained(BASE_MODEL_DIR)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    print("\nBase model training completed.")

# Step 3: Train or load the LoRA model
lora_training_args = create_training_args(output_dir=lora_results_dir, lr=1e-4)
if os.path.exists(LORA_MODEL_DIR):
    print("\nLoRA model already exists. Loading LoRA model...")
    lora_model = AutoModelForSequenceClassification.from_pretrained(LORA_MODEL_DIR)
else:
    print("\nTraining LoRA Model...")
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        #target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS", 
        inference_mode=False,
    )
    # Apply LoRA to model
    lora_model = get_peft_model(base_model, lora_config).to(device)
    trainer_lora = Trainer(
        model=lora_model,
        args=lora_training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print_trainable_params(lora_model, stage_name="LoRA Model")
    start_time = time.time()
    trainer_lora.train()
    print(f"LoRA model training time: {time.time() - start_time}s")
    lora_model.save_pretrained(LORA_MODEL_DIR)
    tokenizer.save_pretrained(LORA_MODEL_DIR)
    print("\nLoRA model training completed.")
    
    
# Step 4: Train or load the Adapter model
adapter_training_args = create_training_args(output_dir=adapter_results_dir, lr=5e-5)
if os.path.exists(ADAPTER_MODEL_DIR):
    print("\nAdapter model already exists. Loading Adapter model...")
    adapter_model = RobertaModel.from_pretrained(ADAPTER_MODEL_DIR)
else:
    print("\nTraining Adapter Model...")
    config = RobertaConfig.from_pretrained(model_name, num_labels=2)

    # Create the custom model
    adapter_model = CustomRobertaModel(config)

    # Load pretrained weights
    pretrained_model = RobertaModel.from_pretrained(model_name)
    adapter_model.load_state_dict(pretrained_model.state_dict(), strict=False)
    adapter_model.apply(initialize_weights)
    
    # Optimizer with layer-wise learning rate decay
    optimizer_grouped_parameters = [
        {"params": [p for n, p in adapter_model.named_parameters() if any(keyword in n for keyword in ["classifier", "down_layer", "up_layer", "layer_norm",])], "lr": 1e-4},
        {"params": [p for n, p in adapter_model.named_parameters() if any(keyword not in n for keyword in ["classifier", "down_layer", "up_layer", "layer_norm",])], "lr": 5e-5},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    
    # Scheduler
    num_training_steps = len(train_dataset) // 16 * num_epochs  # Example for 3 epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    
    trainer_adapter = Trainer(
        model=adapter_model,
        args=adapter_training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    print_trainable_params(adapter_model, stage_name="Adapter Model")
    start_time = time.time()
    trainer_adapter.train()
    print(f"Adapter model training time: {time.time() - start_time}s")
    adapter_model.save_pretrained(ADAPTER_MODEL_DIR)
    tokenizer.save_pretrained(ADAPTER_MODEL_DIR)
    print("\nAdapter model training completed.")


# Step 5: Evaluate all models
print("\nEvaluating all models...")
base_results = evaluate_model(base_model, base_training_args, "Base"
                              )
lora_results = evaluate_model(lora_model, lora_training_args, "LoRA")
adapter_results = evaluate_model(adapter_model, adapter_training_args, "Adapter")

# Summary of results
print("\nSummary of Results:")
print("Base Model:", base_results)
print("LoRA Model:", lora_results)
print("Adapter Model:", adapter_results)