import torch
from causal_self_attention import scaled_dot_product_attention, ScaledDotProductAttentionBackend
import causal_multihead_self_attention

import unittest

_MAX_ERROR = 1e-6


class TestTorchExtension(unittest.TestCase):
    def test_equivalence(self):
        for seq_len in range(1, 200):
            num_heads = 12
            model_dim = 768
            head_dim = model_dim // num_heads
            Q = torch.rand((1, seq_len, model_dim), device="cuda")
            K = torch.rand((1, seq_len, model_dim), device="cuda")
            V = torch.rand((1, seq_len, model_dim), device="cuda")
            cuda_result = scaled_dot_product_attention(ScaledDotProductAttentionBackend.CUSTOM_CUDA, queries=Q, keys=K, values=V, num_heads=num_heads, causal_mask=None)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda"), diagonal=1)
            naive_pytorch_result = scaled_dot_product_attention(ScaledDotProductAttentionBackend.NAIVE_PYTORCH, queries=Q, keys=K, values=V, num_heads=num_heads, causal_mask=causal_mask)
            self.assertLess((cuda_result - naive_pytorch_result).abs().max().item(), _MAX_ERROR)


if __name__ == '__main__':
    unittest.main()
