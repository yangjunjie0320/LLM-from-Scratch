import os, sys, torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

# Our custom dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, block_size=100, max_lines=1000):
        assert os.path.exists(data_path)

        self.enc = GPT2Tokenizer.from_pretrained("gpt2")
        self.block_size = block_size

        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"},
        )[0]

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
    

def train(model, opt_obj, shd_obj, loader, device):
    model.train()
    total_loss = 0

    for ibatch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, targets=y)

        opt_obj.zero_grad()
        loss.backward()
        opt_obj.step()
        shd_obj.step()

        total_loss += loss.item()

        if ibatch % 100 == 0:
            print(f"Epoch {epoch}, Batch {ibatch}, Loss {loss.item()}")

    print(f"Epoch {epoch}, Loss {total_loss / (ibatch + 1)}")
    return total_loss / (ibatch + 1)

def validate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for ibatch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, targets=y)
            total_loss += loss.item()

    return total_loss / (ibatch + 1)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_lines", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    dataset = Dataset(args.data_path, args.block_size, args.max_lines)

    train_set, valid_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    print(f"Training on {device}")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Model has {nparam / 1e6:.2f} M parameters")

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, scheduler, train_loader, device)
        valid_loss = validate(model, valid_loader, device)

        print(f"Epoch {epoch}, Train Loss {train_loss}, Valid Loss {valid_loss}")
