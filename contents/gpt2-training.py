import os, sys, torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

# Our custom dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, block_size):
        assert os.path.exists(data_path)

        self.data = open(data_path, "r", encoding="utf-8").read()
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        