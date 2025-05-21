import torch
from torch import nn

from causal_self_attention import CausalSelfAttention

from feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.first_layer_norm = nn.LayerNorm(config["emb_dim"])
        self.attention = CausalSelfAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            max_context_length=config["context_length"],
            p_dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
            scaled_dot_product_attention_backend=config["scaled_dot_product_attention_backend"],
        )
        self.first_dropout = nn.Dropout(p=config["drop_rate"])
        self.second_layer_norm = nn.LayerNorm(config["emb_dim"])
        self.feed_forward = FeedForward(config)
        self.second_dropout = nn.Dropout(p=config["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.first_layer_norm(x)
        x = self.attention(x)
        x = self.first_dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.second_layer_norm(x)
        x = self.feed_forward(x)
        x = self.second_dropout(x)
        x = x + shortcut

        return x
