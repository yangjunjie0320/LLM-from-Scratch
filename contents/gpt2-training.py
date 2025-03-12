import os, sys
import logging
from datetime import datetime
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from datasets import load_dataset
dataset = load_dataset("json", data_files="/mnt/d/Downloads/pretrain_hq.jsonl")

from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import GPT2Config
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_config = GPT2Config(
    n_positions=512,
    n_embd=384, n_layer=6, n_head=6,
    vocab_size=tokenizer.vocab_size,
    activation_function="gelu_new"
)
model = GPT2LMHeadModel(model_config)
model.resize_token_embeddings(len(tokenizer))

training_config = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

def generate(args, state, control, logs=None, **kwargs):
    if not state.global_step % 1000 == 0:
        return
    
    prompt = "今天天气真好"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated text
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated sample: {text}")

from transformers import TrainerCallback
callback = TrainerCallback()
callback.on_log = generate
    
# Create trainer
trainer = Trainer(
    model=model, args=training_config,
    train_dataset=train_dataset, 
    callbacks=[callback]
)
trainer.train()