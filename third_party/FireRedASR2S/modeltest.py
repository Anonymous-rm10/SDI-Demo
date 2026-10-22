import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.1):
        super().__init__()
        assert d_model % num_heads ==0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model,d_model)
        self.k_proj = nn.Linear(d_model,d_model)
        self.v_proj = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model,d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        q = q.view(
            batch_size,seq_len,self.num_heads,self.head_dim
        ).transpose(1,2)
        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1,2)
        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1,2)
        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if  mask is not None:
            if mask.dim() == 2:
                mask=mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask==0, float="-inf")
        attn_weight = torch.softmax(scores,dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        context = torch.matmul(attn_weight,v)
        context = context.transpose(1,2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output=self.out_proj(context)
        return output

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

model = MultiHeadSelfAttention(d_model, num_heads)

x = torch.randn(batch_size, seq_len, d_model)

output = model(x)

print("input shape:", x.shape)
print("output shape:", output.shape)

assert output.shape == x.shape

print("test passed!")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model,d_model)
        self.k_proh = nn.Linear(d_model,d_model)
        self.v_proj = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model,d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        q = q.view(
            batch_size,seq_len,self.num_heads,self.head_dim
        ).transpose(1,2)
        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1,2)
        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1,2)
        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                mask=mask.unsqueeze(0).unsqueeze(1)
                scores = scores.masked_fill(mask==0, float("-inf"))
            attn_weight = torch.softmax(scores,dim=-1)
            attn_weight = self.attn_dropout(attn_weight)
            context = torch.matmul(attn_weight,v)
            context = context.transpose(1,2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            output=self.out_proj(context)
            return output