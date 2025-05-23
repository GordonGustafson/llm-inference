import torch
from torch import nn
from enum import Enum

# Import custom torch extension that contains
# torch.ops.causal_multihead_self_attention.causal_multihead_self_attention_torch
import causal_multihead_self_attention

class ScaledDotProductAttentionBackend(Enum):
    NAIVE_PYTORCH = "NAIVE_PYTORCH"
    CUSTOM_CUDA = "CUSTOM_CUDA"

def scaled_dot_product_attention(backend: ScaledDotProductAttentionBackend,
                                 queries: torch.Tensor,
                                 keys: torch.Tensor,
                                 values: torch.Tensor,
                                 num_heads: int,
                                 causal_mask: torch.Tensor) -> torch.Tensor:
    match backend:
        case ScaledDotProductAttentionBackend.NAIVE_PYTORCH:
            return scaled_dot_product_attention_naive_pytorch(queries=queries,
                                                              keys=keys,
                                                              values=values,
                                                              num_heads=num_heads,
                                                              causal_mask=causal_mask)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA:
            # TODO: support batch size > 1
            # Custom Cuda backend currently requires contiguous tensors.
            queries = queries.contiguous()
            keys = keys.contiguous()
            values = values.contiguous()
            return torch.ops.causal_multihead_self_attention.causal_multihead_self_attention_torch(Q=queries.squeeze(0),
                                                                                                   K=keys.squeeze(0),
                                                                                                   V=values.squeeze(0),
                                                                                                   num_heads=num_heads).unsqueeze(0)
        case _:
            raise ValueError("Backend not implemented yet")


def scaled_dot_product_attention_naive_pytorch(queries: torch.Tensor,
                                               keys: torch.Tensor,
                                               values: torch.Tensor,
                                               num_heads: int,
                                               causal_mask: torch.Tensor) -> torch.Tensor:
    batch_size, context_length, model_dim = queries.shape
    if model_dim % num_heads != 0:
        raise ValueError("Query dimensionality must be evenly divisible by number of heads.")
    head_dim = model_dim // num_heads
    queries = queries.view(batch_size, context_length, num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, context_length, head_dim)
    keys = keys.view(batch_size, context_length, num_heads, head_dim).transpose(1, 2)        # (batch_size, num_heads, context_length, head_dim)
    values = values.view(batch_size, context_length, num_heads, head_dim).transpose(1, 2)    # (batch_size, num_heads, context_length, head_dim)

    # Contracting along the head_dim axis.
    attention_scores = queries @ keys.transpose(2, 3)  # (batch_size, num_heads, context_length, context_length)
    attention_scores.masked_fill_(causal_mask.bool()[:context_length, :context_length], -torch.inf)
    attention_weights = torch.softmax(attention_scores / (head_dim ** 0.5), dim=3)

    context_vectors = attention_weights @ values      # (batch_size, num_heads, context_length, head_dim)
    context_vectors = context_vectors.transpose(1, 2)  # (batch_size, context_length, num_heads, head_dim)
    context_vectors = context_vectors.contiguous().view(batch_size, context_length, num_heads * head_dim)
    return context_vectors


class CausalSelfAttention(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 max_context_length: int,
                 p_dropout: float,
                 num_heads: int,
                 qkv_bias: bool,
                 scaled_dot_product_attention_backend: ScaledDotProductAttentionBackend):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(f"d_out must be divisible by num_heads. d_out was {d_out} and num_heads was {num_heads}.")

        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.num_heads = num_heads
        self.qkv_layer = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.out_proj_layer = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(p=p_dropout)
        self.scaled_dot_product_attention_backend = scaled_dot_product_attention_backend
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_context_length, max_context_length),
                                                        diagonal=1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, context_length, _ = x.shape
        queries_keys_values = self.qkv_layer(x)
        queries = queries_keys_values[:, :, 0:self.d_out]          # (batch_size, context_length, num_heads * head_dim)
        keys = queries_keys_values[:, :, self.d_out:2*self.d_out]  # (batch_size, context_length, num_heads * head_dim)
        values = queries_keys_values[:, :, 2*self.d_out:]          # (batch_size, context_length, num_heads * head_dim)

        context_vectors = scaled_dot_product_attention(backend=self.scaled_dot_product_attention_backend,
                                                       queries=queries,
                                                       keys=keys,
                                                       values=values,
                                                       num_heads=self.num_heads,
                                                       causal_mask=self.causal_mask)
        context_vectors = self.out_proj_layer(context_vectors)
        return context_vectors

