import os, sys
import logging
from datetime import datetime
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import get_scheduler, set_seed
from transformers import GPT2Tokenizer
from transformers import GPT2Config 
from transformers import GPT2LMHeadModel

# ======== Dataset Class ========
class SimpleTextDataset(Dataset):
    """Simple text dataset class"""
    
    def __init__(self, file_path, tokenizer, block_size=512, max_samples=None):
        """
        Initialize the dataset
        
        Args:
            file_path: Data file path
            tokenizer: Tokenizer
            block_size: Maximum sequence length
            max_samples: Maximum number of samples
        """
        assert os.path.isfile(file_path), f"File does not exist: {file_path}"
        
        print(f"Loading data from file: {file_path}")
        self.examples = []
        
        # Load data
        import json
        lines = []
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                try:
                    line_data = json.loads(line.strip())
                    if isinstance(line_data, dict) and "text" in line_data:
                        lines.append(line_data["text"])
                    else:
                        lines.append(line.strip())
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text
                    lines.append(line.strip())
        
        # Tokenize all texts
        tokenized_text = []
        for text in lines:
            tokenized_text.extend(tokenizer.encode(text))
            # Add EOS token after each text
            tokenized_text.append(tokenizer.eos_token_id)
        
        # Create samples of length block_size
        for i in range(0, len(tokenized_text) - block_size, block_size):
            self.examples.append(tokenized_text[i:i + block_size])
        
        print(f"Created {len(self.examples)} samples from {len(lines)} texts")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


# ======== Training Configuration Class ========
class TrainingConfig:
    """Configuration class for GPT-2 training"""
    block_size = 512
    max_samples = 1000
    num_train_epochs = 20
    batch_size = 4
    learning_rate = 5e-5
    seed = 42
    logging_steps = 200
    
    def __init__(self, model=None, tokenizer=None, dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Adjust token embedding size
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Log model size
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {num_params / 1e6:.2f} million parameters")
    
    def generate_sample_text(self, prompt="今天天气真好", max_length=100):
        """Generate sample text to monitor training progress"""
        
        if self.model is None or self.tokenizer is None or self.device is None:
            raise ValueError("Model, tokenizer, and device must be set before generating text")
            
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.8,
            top_k=50, top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
        )
        
        # Decode and return generated text
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text
    
    def run(self):
        """Main training function"""
        
        # Set random seed
        set_seed(self.seed)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"{self.output_dir}_{self.model_type}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to {output_dir}")
        
        model = self.model
        param = model.parameters()
        tokenizer = self.tokenizer
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Calculate total training steps
        total_steps = len(dataloader) * self.num_train_epochs
        
        # Create optimizer
        optimizer = AdamW(
            param, eps=1e-8, 
            lr=self.learning_rate
        )

        scheduler = get_scheduler(
            name="linear", optimizer=optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # Training loop
        print("***** Starting training *****")
        print(f"  Number of samples = {len(self.dataset)}")
        print(f"  Number of epochs = {self.num_train_epochs}")
        print(f"  Batch size = {self.batch_size}")
        print(f"  Total steps = {total_steps}")
        print(f"  Learning rate = {self.learning_rate}")
        
        global_step = 0
        loss_tot = 0.0
        
        model.zero_grad()
        
        # Training loop
        num_train_epochs = self.num_train_epochs
        logging_steps = self.logging_steps

        for epoch in range(num_train_epochs):
            epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1:4d}/{num_train_epochs:4d}")
            
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch, labels=batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                loss_tot += loss.item()
                global_step += 1
                
                # Update weights
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                # Log progress
                if global_step % logging_steps == 0:
                    perplexity = torch.exp(torch.tensor(loss_tot / global_step))
                    loss_avg = loss_tot / global_step

                    # Log information
                    print(f"Step {global_step}/{total_steps} - Loss: {loss_avg:6.2e} - Perplexity: {perplexity:6.2e}")
                    
                    # Generate sample text
                    sample_text = self.generate_sample_text()
                    print(f"Generated sample: {sample_text}")

                    self.save_model(global_step)

        self.save_model()
        print(f"Training completed, total {global_step} steps")

if __name__ == "__main__":
    """Main function to run training"""
    from datasets import load_dataset
    dataset = load_dataset("pretrain_hq.jsonl")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_config = GPT2Config(
        n_positions=512,
        n_embd=384, n_layer=6, n_head=6,
        vocab_size=tokenizer.vocab_size,
        activation_function="gelu_new"
    )
    model = GPT2LMHeadModel.from_pretrained("gpt2", model_config)
    model.resize_token_embeddings(len(tokenizer))

    from transformers import Trainer, TrainingArguments
    from transformers import GPT2LMHeadModel
    from transformers import GPT2Tokenizer
    from transformers import GPT2Config

    training_config = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        per_device_train_batch_size=4,
    )
    trainer = Trainer(
        model=model, args=training_config,
        train_dataset=dataset,
    )
    trainer.train()