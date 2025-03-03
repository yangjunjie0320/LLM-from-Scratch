import os, sys, torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import time
from datetime import datetime, timedelta
import warnings
import json
import argparse

# Suppress specific HuggingFace warnings
warnings.filterwarnings("ignore", message=".*Keyword arguments.*not recognized.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")
warnings.filterwarnings("ignore", message=".*The model.*with a 'ForCausalLM'.*")

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

# Our custom dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, block_size=100, max_lines=1000):
        assert os.path.exists(data_path)

        # Initialize tokenizer with proper settings
        self.enc = GPT2Tokenizer.from_pretrained("gpt2")
        self.block_size = block_size

        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Get the endoftext token - no warnings with this approach
        self.eos_token = self.enc.eos_token_id
        
        # If eos_token_id is not available, fall back to manual encoding
        if self.eos_token is None:
            self.eos_token = self.enc.convert_tokens_to_ids(self.enc.eos_token or "<|endoftext|>")

        import json
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for iline, line in enumerate(f):
                if iline >= max_lines:
                    break

                line = json.loads(line.strip())['text']
                line = self.enc.encode(line)
                data.extend(line + [self.eos_token])

        data_len = len(data)

        self.data = []
        for i in range(0, data_len, block_size):
            chunk = data[i:i+block_size]
            chunk += [self.eos_token] * (block_size - len(chunk))
            self.data.append(chunk)

        print(f"Loaded {len(self.data)} chunks from {data_path}")
        print(f"Each chunk has {block_size} tokens")
        print(f"Total tokens: {sum(len(chunk) for chunk in self.data)}")
        print(f"Total lines: {iline + 1}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, ids):
        return self.enc.decode(ids)


def train(model, opt_obj, shd_obj, loader, device, epoch):
    model.train()
    total_loss = 0
    criterion = CrossEntropyLoss()
    
    # Track timing and tokens
    start_time = time.time()
    total_tokens = 0
    
    # Get total batches for progress tracking
    total_batches = len(loader)

    for ibatch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
        # Count tokens processed
        batch_tokens = x.numel()
        total_tokens += batch_tokens

        # Fix the model call and loss calculation
        outputs = model(x, labels=y)
        loss = outputs.loss

        opt_obj.zero_grad()
        loss.backward()
        opt_obj.step()
        shd_obj.step()

        total_loss += loss.item()

        # Calculate elapsed time and tokens per second
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        
        # Get current learning rate
        current_lr = opt_obj.param_groups[0]['lr']
        
        # Create progress indicator
        progress = (ibatch + 1) / total_batches
        progress_bar = '=' * int(30 * progress) + '>' + ' ' * (30 - int(30 * progress))
        
        # Estimate time remaining
        if tokens_per_sec > 0:
            tokens_remaining = (total_batches - ibatch - 1) * batch_tokens
            time_remaining = tokens_remaining / tokens_per_sec
            eta = datetime.now() + timedelta(seconds=time_remaining)
            eta_str = eta.strftime("%H:%M:%S")
        else:
            eta_str = "Unknown"

        if ibatch % 10 == 0:  # More frequent updates (every 10 batches)
            print(f"\rEpoch {epoch} [{progress_bar}] {ibatch+1}/{total_batches} ({progress:.1%}) "
                  f"Loss: {loss.item():.4f} (Avg: {total_loss/(ibatch+1):.4f}) "
                  f"LR: {current_lr:.6f} "
                  f"Speed: {tokens_per_sec:.1f} tokens/s "
                  f"ETA: {eta_str}", end="")
            sys.stdout.flush()

    # Calculate final metrics
    epoch_loss = total_loss / total_batches
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    
    # Print summary for the epoch
    print(f"\nEpoch {epoch} completed in {elapsed:.2f}s - "
          f"Loss: {epoch_loss:.4f} - "
          f"Speed: {tokens_per_sec:.1f} tokens/s - "
          f"Learning rate: {current_lr:.6f}")
    
    return epoch_loss


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    start_time = time.time()
    total_tokens = 0
    total_batches = len(loader)
    
    # Progress tracking
    print(f"Validating: ", end="")

    with torch.no_grad():
        for ibatch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            
            total_tokens += x.numel()

            outputs = model(x, labels=y)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            # Update progress
            if ibatch % 10 == 0:
                progress = (ibatch + 1) / total_batches
                progress_bar = '=' * int(20 * progress) + '>' + ' ' * (20 - int(20 * progress))
                print(f"\rValidating: [{progress_bar}] {ibatch+1}/{total_batches} ({progress:.1%})", end="")
                sys.stdout.flush()
    
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    valid_loss = total_loss / total_batches
    
    print(f"\rValidation completed in {elapsed:.2f}s - Loss: {valid_loss:.4f} - Speed: {tokens_per_sec:.1f} tokens/s")
    
    return valid_loss


def generate_sample(model, tokenizer, device, prompt="请告诉我世界上最高的山峰是哪座？", max_length=50):
    """Generate a text sample from the model during training to monitor progress."""
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def parse_arguments():



def print_config(args):
    """Print training configuration."""
    print("\n" + "="*50)
    print(f"GPT-2 Training Configuration")
    print("="*50)
    print(f"Data path:       {args.data_path}")
    print(f"Block size:      {args.block_size}")
    print(f"Max lines:       {args.max_lines}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"Save directory:  {args.save_dir}")
    print(f"Eval interval:   {args.eval_interval}")
    print(f"Generate samples: {args.generate_samples}")
    print("="*50 + "\n")


def prepare_dataset(args):
    """Prepare dataset and dataloaders."""
    dataset = Dataset(args.data_path, args.block_size, args.max_lines)
    
    # Print dataset information after loading
    print(f"Dataset loaded with {len(dataset)} chunks")
    
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    print(f"Dataset split: {train_size} training chunks, {valid_size} validation chunks")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    
    return dataset, train_loader, valid_loader


def initialize_model(device):
    """Initialize the GPT-2 model."""
    # Create a custom config without the problematic loss_type parameter
    config = GPT2Config.from_pretrained("gpt2")
    
    # Remove the loss_type attribute if it exists in the config
    if hasattr(config, 'loss_type'):
        delattr(config, 'loss_type')
    
    # Load model with custom config
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)

    print(f"Training on {device}")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Model has {nparam / 1e6:.2f} M parameters")
    
    return model


def setup_training(model, args):
    """Setup optimizer and scheduler."""
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, valid_loss, output_dir):
    """Save a model checkpoint."""
    checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def save_final_model(model, output_dir):
    """Save the final model."""
    final_model_path = f"{output_dir}/final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def save_training_history(train_losses, valid_losses, args, training_time, output_dir):
    """Save training history to a JSON file."""
    history = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'args': vars(args),
        'training_time': training_time,
    }
    
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Training history saved to {output_dir}/training_history.json")


def train_model(model, optimizer, scheduler, train_loader, valid_loader, 
                device, args, output_dir, dataset):
    """Main training loop."""
    # Store training metrics for plotting later
    train_losses = []
    valid_losses = []
    
    # Track overall training time
    training_start_time = time.time()
    
    print("\nStarting training...")
    try:
        for epoch in range(args.epochs):
            # Training phase
            train_loss = train(model, optimizer, scheduler, train_loader, device, epoch)
            
            # Validation phase (only run every eval_interval epochs)
            if (epoch % args.eval_interval) == 0 or epoch == args.epochs - 1:
                valid_loss = validate(model, valid_loader, device)
            else:
                valid_loss = None
                
            # Store metrics
            train_losses.append(train_loss)
            if valid_loss is not None:
                valid_losses.append(valid_loss)
            
            # Print epoch summary
            if valid_loss is not None:
                print(f"Epoch {epoch} summary - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            else:
                print(f"Epoch {epoch} summary - Train Loss: {train_loss:.4f}")
            
            # Save checkpoint
            if valid_loss is not None:
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss, valid_loss, output_dir)
            
            # Generate sample text if requested
            if args.generate_samples and hasattr(dataset, 'enc'):
                sample = generate_sample(model, dataset.enc, device)
                print(f"Sample generation: {sample}")
            
            print("-" * 50)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model...")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    print(f"\nTraining completed in {training_time_str}")
    
    # Print final metrics if available
    if train_losses:
        print(f"Final training loss: {train_losses[-1]:.4f}")
    if valid_losses:
        print(f"Final validation loss: {valid_losses[-1]:.4f}")
    
    # Save final model
    save_final_model(model, output_dir)
    
    # Save training history
    save_training_history(train_losses, valid_losses, args, training_time_str, output_dir)
    
    return train_losses, valid_losses


def main(config):
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"gpt2_{timestamp}"
    output_dir = config.save_dir if config.save_dir else f"output_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to {output_dir}/")
    
    # Print configuration
    print_config(config)
    
    # Prepare dataset and dataloaders
    dataset, train_loader, valid_loader = prepare_dataset(config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = initialize_model(device)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_training(model, config)
    
    # Train the model
    train_losses, valid_losses = train_model(
        model, optimizer, scheduler, train_loader, valid_loader, 
        device, config, output_dir, dataset
    )
    
    return train_losses, valid_losses


if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a GPT-2 model on custom data")
    parser.add_argument("--data_path", type=str, default="data/train.jsonl",
                        help="Path to the training data file")
    parser.add_argument("--block_size", type=int, default=512,
                        help="Size of token blocks for training")
    parser.add_argument("--max_lines", type=int, default=500,
                        help="Maximum number of lines to read from the data file")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save model checkpoints (default: auto-generated)")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Interval (in epochs) to run evaluation")
    parser.add_argument("--generate_samples", action="store_true",
                        help="Generate text samples during training")
    config = parser.parse_args()
    main(config)
