import torch
from torch import nn
from enum import Enum


class ScaledDotProductAttentionBackend(Enum):
    NAIVE_PYTORCH = "NAIVE_PYTORCH"
    CUSTOM_CUDA = "CUSTOM_CUDA"
    PYTORCH_SDPA_MATH = "PYTORCH_SDPA_MATH"
    PYTORCH_SDPA_FLASH_ATTENTION = "PYTORCH_SDPA_FLASH_ATTENTION"
    PYTORCH_SDPA_EFFICIENT_ATTENTION = "PYTORCH_SDPA_EFFICIENT_ATTENTION"
    PYTORCH_SDPA_CUDNN_ATTENTION = "PYTORCH_SDPA_CUDNN_ATTENTION"

    # Specific versions of the custom cuda implementation.
    CUSTOM_CUDA_VERSION_1  = "CUSTOM_CUDA_VERSION_1"
    CUSTOM_CUDA_VERSION_2  = "CUSTOM_CUDA_VERSION_2"
    CUSTOM_CUDA_VERSION_3  = "CUSTOM_CUDA_VERSION_3"
    CUSTOM_CUDA_VERSION_4  = "CUSTOM_CUDA_VERSION_4"
    CUSTOM_CUDA_VERSION_5  = "CUSTOM_CUDA_VERSION_5"
    CUSTOM_CUDA_VERSION_6  = "CUSTOM_CUDA_VERSION_6"
    CUSTOM_CUDA_VERSION_7  = "CUSTOM_CUDA_VERSION_7"
    CUSTOM_CUDA_VERSION_8  = "CUSTOM_CUDA_VERSION_8"
    CUSTOM_CUDA_VERSION_9  = "CUSTOM_CUDA_VERSION_9"
    CUSTOM_CUDA_VERSION_10 = "CUSTOM_CUDA_VERSION_10"

PYTORCH_SDPA_BACKENDS = [ScaledDotProductAttentionBackend.PYTORCH_SDPA_MATH,
                         ScaledDotProductAttentionBackend.PYTORCH_SDPA_FLASH_ATTENTION,
                         ScaledDotProductAttentionBackend.PYTORCH_SDPA_EFFICIENT_ATTENTION,
                         ScaledDotProductAttentionBackend.PYTORCH_SDPA_CUDNN_ATTENTION]

ALL_VERSIONED_CUSTOM_CUDA_BACKENDS = [ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_1,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_2,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_3,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_4,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_5,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_6,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_7,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_8,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_9,
                                      ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_10]


def load_attention_backend(backend: ScaledDotProductAttentionBackend) -> None:
    match backend:
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA:
            # Import custom torch extension that contains
            # torch.ops.causal_multihead_self_attention.causal_multihead_self_attention_torch
            import causal_multihead_self_attention
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_1:
            # Import custom torch extension that contains
            # torch.ops.causal_multihead_self_attention_version_1.causal_multihead_self_attention_torch
            import causal_multihead_self_attention_version_1
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_2:
            # etc...
            import causal_multihead_self_attention_version_2
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_3:
            import causal_multihead_self_attention_version_3
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_4:
            import causal_multihead_self_attention_version_4
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_5:
            import causal_multihead_self_attention_version_5
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_6:
            import causal_multihead_self_attention_version_6
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_7:
            import causal_multihead_self_attention_version_7
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_8:
            import causal_multihead_self_attention_version_8
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_9:
            import causal_multihead_self_attention_version_9
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_10:
            import causal_multihead_self_attention_version_10
        case _:
            pass


def scaled_dot_product_attention(backend: ScaledDotProductAttentionBackend,
                                 queries: torch.Tensor,
                                 keys: torch.Tensor,
                                 values: torch.Tensor,
                                 num_heads: int,
                                 causal_mask: torch.Tensor) -> torch.Tensor:
    if (backend == ScaledDotProductAttentionBackend.CUSTOM_CUDA
        or backend in ALL_VERSIONED_CUSTOM_CUDA_BACKENDS):
        # TODO: support batch size > 1
        # Custom Cuda backend currently requires contiguous tensors.
        queries = queries.contiguous().squeeze(0)
        keys = keys.contiguous().squeeze(0)
        values = values.contiguous().squeeze(0)

    if backend in PYTORCH_SDPA_BACKENDS:
        batch_size, context_length, model_dim = queries.shape
        head_dim = model_dim // num_heads
        queries = queries.view(batch_size, context_length, num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, context_length, head_dim)
        keys = keys.view(batch_size, context_length, num_heads, head_dim).transpose(1, 2)        # (batch_size, num_heads, context_length, head_dim)
        values = values.view(batch_size, context_length, num_heads, head_dim).transpose(1, 2)    # (batch_size, num_heads, context_length, head_dim)

    match backend:
        case ScaledDotProductAttentionBackend.NAIVE_PYTORCH:
            return scaled_dot_product_attention_naive_pytorch(queries=queries,
                                                              keys=keys,
                                                              values=values,
                                                              num_heads=num_heads,
                                                              causal_mask=causal_mask)
        case ScaledDotProductAttentionBackend.PYTORCH_SDPA_MATH:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                result = torch.nn.functional.scaled_dot_product_attention(query=queries, key=keys, value=values, is_causal=True)
            return result.transpose(1, 2).view(batch_size, context_length, model_dim)
        case ScaledDotProductAttentionBackend.PYTORCH_SDPA_FLASH_ATTENTION:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                result = torch.nn.functional.scaled_dot_product_attention(query=queries, key=keys, value=values, is_causal=True)
            return result.transpose(1, 2).view(batch_size, context_length, model_dim)
        case ScaledDotProductAttentionBackend.PYTORCH_SDPA_EFFICIENT_ATTENTION:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                result = torch.nn.functional.scaled_dot_product_attention(query=queries, key=keys, value=values, is_causal=True)
            return result.transpose(1, 2).view(batch_size, context_length, model_dim)
        case ScaledDotProductAttentionBackend.PYTORCH_SDPA_CUDNN_ATTENTION:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                result = torch.nn.functional.scaled_dot_product_attention(query=queries, key=keys, value=values, is_causal=True)
            return result.transpose(1, 2).view(batch_size, context_length, model_dim)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA:
            return torch.ops.causal_multihead_self_attention.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_1:
            return torch.ops.causal_multihead_self_attention_version_1.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_2:
            return torch.ops.causal_multihead_self_attention_version_2.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_3:
            return torch.ops.causal_multihead_self_attention_version_3.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_4:
            return torch.ops.causal_multihead_self_attention_version_4.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_5:
            return torch.ops.causal_multihead_self_attention_version_5.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_6:
            return torch.ops.causal_multihead_self_attention_version_6.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_7:
            return torch.ops.causal_multihead_self_attention_version_7.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_8:
            return torch.ops.causal_multihead_self_attention_version_8.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_9:
            return torch.ops.causal_multihead_self_attention_version_9.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
        case ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_10:
            return torch.ops.causal_multihead_self_attention_version_10.causal_multihead_self_attention_torch(Q=queries, K=keys, V=values, num_heads=num_heads).unsqueeze(0)
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
        load_attention_backend(scaled_dot_product_attention_backend)

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

