import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, max_context_length: int, p_dropout: float, num_heads: int, qkv_bias: bool):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(f"d_out must be divisible by num_heads. d_out was {d_out} and num_heads was {num_heads}.")

        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.num_heads = num_heads
        self.qkv_layer = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.out_proj_layer = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(p=p_dropout)
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_context_length, max_context_length),
                                                       diagonal=1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, context_length, _ = x.shape
        queries_keys_values = self.qkv_layer(x)
        queries = queries_keys_values[:, :, 0:self.d_out]          # (batch_size, context_length, num_heads * head_dim)
        keys = queries_keys_values[:, :, self.d_out:2*self.d_out]  # (batch_size, context_length, num_heads * head_dim)
        values = queries_keys_values[:, :, 2*self.d_out:]          # (batch_size, context_length, num_heads * head_dim)

        queries = queries.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, context_length, head_dim)
        keys = keys.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)        # (batch_size, num_heads, context_length, head_dim)
        values = values.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, context_length, head_dim)

        # Contracting along the head_dim axis.
        attention_scores = queries @ keys.transpose(2, 3)  # (batch_size, num_heads, context_length, context_length)
        attention_scores.masked_fill_(self.causal_mask.bool()[:context_length, :context_length], -torch.inf)
        attention_weights = torch.softmax(attention_scores / (self.head_dim ** 0.5), dim=3)

        context_vectors = attention_weights @ values      # (batch_size, num_heads, context_length, head_dim)
        context_vectors = context_vectors.transpose(1, 2)  # (batch_size, context_length, num_heads, head_dim)
        context_vectors = context_vectors.contiguous().view(batch_size, context_length, self.num_heads * self.head_dim)
        context_vectors = self.out_proj_layer(context_vectors)
        return context_vectors



