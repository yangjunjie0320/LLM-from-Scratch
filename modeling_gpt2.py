from math import sqrt, pi

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from dataclasses import dataclass

class Conv1D(nn.Module):
    """
    1D-convolutional layer as used in OpenAI GPT.

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        nx, nf = self.nx, self.nf

        shape_inp = x.shape
        shape_out = shape_inp[:-1] + (nf,)

        w = self.weight
        b = self.bias
        x = torch.einsum("xi,io->xo", x.view(-1, nx), w)
        y = x + b
        y = y.view(shape_out)
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=None, n_head=None, block_size=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = Conv1D(3 * n_embd, n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)

        # regularization
        self.block_size = block_size
        self.n_head = n_head
        self.n_embd = n_embd

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        ones = torch.ones(block_size, block_size)
        ones = ones.view(1, 1, block_size, block_size)
        self.register_buffer("bias", torch.tril(ones))

    def forward(self, x):
        n_embd = self.n_embd
        n_head = self.n_head

        # d stands for the dimension per head
        d = n_embd // n_head 
        assert d * n_head == n_embd

        batch_size, seq_len = x.shape[:2]
        assert x.shape == (batch_size, seq_len, n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x) # (..., n_embd) -> (..., 3 * n_embd)
        assert qkv.shape == (batch_size, seq_len, 3 * n_embd)
        
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(batch_size, seq_len, n_head, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, n_head, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, n_head, -1).transpose(1, 2)

        assert q.shape == (batch_size, n_head, seq_len, d)
        assert k.shape == (batch_size, n_head, seq_len, d)
        assert v.shape == (batch_size, n_head, seq_len, d)

        # the last dimension is d, which is contracted
        qk = torch.einsum("bhqd,bhkd->bhqk", q, k)
        qk = qk / sqrt(d)
        assert qk.shape == (batch_size, n_head, seq_len, seq_len)

        bias = self.bias[:, :, :seq_len, :seq_len]
        assert bias.shape == (1, 1, seq_len, seq_len)

        from torch.nn.functional import softmax
        qk = qk.masked_fill(bias == 0, float('-inf'))
        qk = softmax(qk, dim=-1)

        x = torch.einsum("bhqk,bhkd->bhqd", qk, v)
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, n_embd)

        y = self.c_proj(x)
        assert y.shape == (batch_size, seq_len, n_embd)
        return y
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config.vocab_size
        n_embd = config.n_embd
        n_layer = config.n_layer
        n_head = config.n_head
        block_size = config.max_position_embeddings

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size

        transformer = {}
        h = [Block(config) for _ in range(n_layer)]
        transformer["h"]    = nn.ModuleList(h)
        transformer["wte"]  = nn.Embedding(vocab_size, n_embd)
        transformer["wpe"]  = nn.Embedding(block_size, n_embd)
        transformer["ln_f"] = nn.LayerNorm(n_embd)
        self.transformer = nn.ModuleDict(transformer)

        # lm_head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

class GeLU(nn.Module):
    def forward(self, x):
        y = x + 0.044715 * torch.pow(x, 3.0)
        y *= sqrt(2.0 / pi)
        y = 0.5 * x * (1.0 + torch.tanh(y))
        return y

class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_embd=None, n_imds=None):
        super().__init__()
        self.c_fc   = Conv1D(n_imds, n_embd)
        self.c_proj = Conv1D(n_embd, n_imds)
        self.act   = GeLU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        y = self.c_proj(x)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_embd = config.n_embd
        n_imds = config.n_inner if config.n_inner is not None else 4 * n_embd

        n_head = config.n_head
        block_size = config.max_position_embeddings

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MultiLayerPerceptron(n_embd=n_embd, n_imds=n_imds)

    def forward(self, x):
        y = self.ln_1(x)
        y = x + self.attn(y)

        x = y
        y = self.ln_2(x)
        y = x + self.mlp(y)
        return y

from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers.generation import GenerationMixin
class GPT(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        vocab_size = config.vocab_size
        n_embd = config.n_embd
        n_layer = config.n_layer
        n_head = config.n_head
        block_size = config.max_position_embeddings

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size

        transformer = {}
        h = [Block(config) for _ in range(n_layer)]
        transformer["h"]    = nn.ModuleList(h)
        transformer["wte"]  = nn.Embedding(vocab_size, n_embd)
        transformer["wpe"]  = nn.Embedding(block_size, n_embd)
        transformer["ln_f"] = nn.LayerNorm(n_embd)
        self.transformer = nn.ModuleDict(transformer)

        # lm_head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, input_ids, return_dict=None, **kwargs):
        device = input_ids.device

        # idx is of shape (batch_size, seq_len), with integers
        # in the range of [0, vocab_size)
        n_embd = self.n_embd
        batch_size, seq_len = input_ids.size()

        block_size = self.block_size
        info = f"Cannot forward sequence of length {seq_len}, block size is only {block_size}"
        assert seq_len <= block_size, info
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device) # shape (seq_len)

        # forward the GPT model itself
        idx = input_ids
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        assert tok_emb.shape == (batch_size, seq_len, n_embd)
        assert pos_emb.shape == (seq_len, n_embd)

        x = tok_emb + pos_emb[None, :, :]
        assert x.shape == (batch_size, seq_len, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # not evaluating loss
        y = self.lm_head(x)
        g = None

        if not return_dict:
            return y

        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        res = CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=y,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
        return res

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        from transformers import GPT2Config
        config = GPT2Config.from_pretrained(model_type)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model
    
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    text = "My name is"
    tokens = tokenizer.encode(text, return_tensors="pt")

    m_sol = GPT.from_pretrained("gpt2")
    m_sol.eval()
    y_sol = m_sol.forward(tokens)[0]
    print(y_sol.shape, y_sol.device, y_sol.dtype)

    from transformers import GPT2LMHeadModel
    m_ref = GPT2LMHeadModel.from_pretrained("gpt2")
    m_ref.eval()
    y_ref = m_ref.forward(tokens)[0]

    err = abs(y_sol - y_ref).max()
    print("err = %6.4e" % err)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inp = tokenizer(text, return_tensors="pt", padding=True)

    m_sol.eval()
    out_sol = m_sol.generate(
        **inp, max_new_tokens=20, 
        pad_token_id=tokenizer.eos_token_id, 
        temperature=1.0, repetition_penalty=1.2,
        do_sample=True,
    )
    print(tokenizer.decode(out_sol[0].tolist()).replace("\n", ""))

    out_ref = m_ref.generate(
        **inp, max_new_tokens=20, 
        pad_token_id=tokenizer.eos_token_id, 
        temperature=1.0, repetition_penalty=1.2,
        do_sample=True,
    )
    print(tokenizer.decode(out_ref[0].tolist()).replace("\n", ""))
