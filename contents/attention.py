import torch
from torch import nn
from torch.nn.functional import softmax

from numpy import sqrt

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.inp_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        embed_dim = self.embed_dim
        batch_size, seq_len = x.shape[:2]
        assert x.shape == (batch_size, seq_len, embed_dim)

        inp = self.inp_proj(x)
        que, key, val = inp.chunk(3, dim=-1)

        attn_score = torch.einsum("bsm,brm->bsr", que, key)
        attn_score *= 1.0 / sqrt(embed_dim)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float('-inf'))

        # for single-head attention, there is no need 
        # to introduce out_proj 
        attn_weigh = softmax(attn_score, dim=-1)
        attn = torch.einsum("bsr,brm->bsm", attn_weigh, val)
        y = self.out_proj(attn)
        return y
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        embed_dim_per_head = embed_dim // num_heads
        assert embed_dim_per_head * num_heads == embed_dim

        self.inp_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        embed_dim_per_head = embed_dim // num_heads

        batch_size, seq_len = x.shape[:2]
        assert x.shape == (batch_size, seq_len, embed_dim)

        inp = self.inp_proj(x)
        que, key, val = inp.chunk(3, dim=-1)
        que = que.view(batch_size, seq_len, num_heads, embed_dim_per_head)
        key = key.view(batch_size, seq_len, num_heads, embed_dim_per_head)
        val = val.view(batch_size, seq_len, num_heads, embed_dim_per_head)

        attn_score = torch.einsum("bshd,brhd->bhsr", que, key)
        attn_score *= 1.0 / sqrt(embed_dim_per_head)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float('-inf'))

        attn_weigh = softmax(attn_score, dim=-1)
        attn = torch.einsum("bhsr,brhd->bshd", attn_weigh, val)
        attn = attn.contiguous().view(batch_size, seq_len, embed_dim)
        y = self.out_proj(attn)
        return y
    
def check_sha(m: SingleHeadAttention):
    m_sol = m
    embed_dim = m_sol.embed_dim
    
    # Create PyTorch's MultiheadAttention with just one head for comparison
    m_ref = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)

    sd_ref = m_ref.state_dict()
    sd_ref["in_proj_weight"] = m_sol.inp_proj.weight
    sd_ref["in_proj_bias"] = m_sol.inp_proj.bias
    sd_ref["out_proj.weight"] = m_sol.out_proj.weight
    sd_ref["out_proj.bias"] = m_sol.out_proj.bias
    m_ref.load_state_dict(sd_ref)

    # Create test inputs
    batch_size = 10
    seq_len = 100
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    x = torch.randn(batch_size, seq_len, embed_dim)
    y_sol = m_sol(x, mask=attn_mask)
    y_ref = m_ref(x, x, x, attn_mask=attn_mask)[0]
    assert abs(y_sol - y_ref).max() < 1e-5

def check_ma(m: MultiHeadAttention):
    m_sol = m
    embed_dim = m_sol.embed_dim
    num_heads = m_sol.num_heads
    embed_dim_per_head = embed_dim // num_heads
    assert embed_dim_per_head * num_heads == embed_dim

    # Create PyTorch's MultiheadAttention with just one head for comparison
    m_ref = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    sd_ref = m_ref.state_dict()
    sd_ref["in_proj_weight"] = m_sol.inp_proj.weight
    sd_ref["in_proj_bias"] = m_sol.inp_proj.bias
    sd_ref["out_proj.weight"] = m_sol.out_proj.weight
    sd_ref["out_proj.bias"] = m_sol.out_proj.bias
    m_ref.load_state_dict(sd_ref)

    batch_size = 10
    seq_len = 100
    attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    x = torch.randn(batch_size, seq_len, embed_dim)
    y_sol = m_sol(x, mask=attn_mask)
    y_ref = m_ref(x, x, x, attn_mask=attn_mask)[0]
    assert abs(y_sol - y_ref).max() < 1e-5

if __name__ == "__main__":
    sha = SingleHeadAttention(embed_dim=200)
    check_sha(sha)

    ma = MultiHeadAttention(embed_dim=200, num_heads=10)
    check_ma(ma)
