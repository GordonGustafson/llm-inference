from sampler import GreedySampler
from transformer_block import TransformerBlock
from causal_self_attention import ScaledDotProductAttentionBackend
from tokenizer import GPT2Tokenizer

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors_utils import assign_weight_and_bias
import torch
from torch import nn

import time

def _assign_transformer_block(block: TransformerBlock, safetensors_object: safe_open, block_index: int) -> None:
    assign_weight_and_bias(block.first_layer_norm, safetensors_object, f"h.{block_index}.ln_1")

    assign_weight_and_bias(block.attention.qkv_layer, safetensors_object, f"h.{block_index}.attn.c_attn", transpose_weight=True)
    assign_weight_and_bias(block.attention.out_proj_layer, safetensors_object, f"h.{block_index}.attn.c_proj", transpose_weight=True)

    assign_weight_and_bias(block.second_layer_norm, safetensors_object, f"h.{block_index}.ln_2")

    assign_weight_and_bias(block.feed_forward.first_layer, safetensors_object, f"h.{block_index}.mlp.c_fc", transpose_weight=True)
    assign_weight_and_bias(block.feed_forward.second_layer, safetensors_object, f"h.{block_index}.mlp.c_proj", transpose_weight=True)


class GPT2(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=config["vocab_size"],
                                             embedding_dim=config["emb_dim"])
        self.positional_embeddings = nn.Embedding(num_embeddings=config["context_length"],
                                                  embedding_dim=config["emb_dim"])
        self.dropout = nn.Dropout(p=config["drop_rate"])
        self.transformer_blocks = [TransformerBlock(config) for _ in range(config["n_layers"])]
        self.layer_norm = nn.LayerNorm(config["emb_dim"])
        self.output_layer = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        self.layers = nn.Sequential(*[self.dropout, *self.transformer_blocks, self.layer_norm, self.output_layer])

    def forward(self, tokens: torch.Tensor):
        batch_size, sequence_length = tokens.shape
        x = self.token_embeddings(tokens)
        x = x + self.positional_embeddings(torch.arange(0, sequence_length, device=tokens.device))
        return self.layers(x)

    def load_weights_from_huggingface(self, repo_id: str) -> None:
        safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        with (safe_open(safetensors_path, framework="pt") as f,
              torch.no_grad()):
            assign_weight_and_bias(self.token_embeddings, f, "wte")
            assign_weight_and_bias(self.positional_embeddings, f, "wpe")
            for i, block in enumerate(self.transformer_blocks):
                _assign_transformer_block(block=block, safetensors_object=f, block_index=i)
            assign_weight_and_bias(self.layer_norm, f, "ln_f")
            assign_weight_and_bias(self.output_layer, f, "wte")



if __name__ == "__main__":
    _COMMON_GPT_CONFIG = {
        "scaled_dot_product_attention_backend": ScaledDotProductAttentionBackend.CUSTOM_CUDA,
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias
    }
    HF_NAME_TO_GPT_CONFIG = {
        "openai-community/gpt2":        {**_COMMON_GPT_CONFIG, **{"emb_dim": 768,  "n_heads": 12, "n_layers": 12}},
        "openai-community/gpt2-medium": {**_COMMON_GPT_CONFIG, **{"emb_dim": 1024, "n_heads": 16, "n_layers": 24}},
        "openai-community/gpt2-large":  {**_COMMON_GPT_CONFIG, **{"emb_dim": 1280, "n_heads": 20, "n_layers": 36}},
        "openai-community/gpt2-xl":     {**_COMMON_GPT_CONFIG, **{"emb_dim": 1600, "n_heads": 25, "n_layers": 48}},
    }

    hf_name = "openai-community/gpt2"
    model = GPT2(HF_NAME_TO_GPT_CONFIG[hf_name])
    model.eval()
    model.load_weights_from_huggingface(hf_name)
    device = torch.device("cuda")
    model.to(device)

    tokenizer = GPT2Tokenizer()
    sampler = GreedySampler()
    torch.manual_seed(0)

    print("Executing first run...")
    start_time_ns = time.perf_counter_ns()
    result = sampler.sample(model, "Hello, I'm a language model,", 500, tokenizer, device)
    end_time_ns = time.perf_counter_ns()
    print(result)
    print(f"time elapsed by first run: {(end_time_ns - start_time_ns) / (10 ** 9)} seconds")

    print("Executing second run...")
    start_time_ns = time.perf_counter_ns()
    result = sampler.sample(model, "Hello, I'm a language model,", 500, tokenizer, device)
    end_time_ns = time.perf_counter_ns()
    print(result)
    print(f"time elapsed by second run: {(end_time_ns - start_time_ns) / (10 ** 9)} seconds")

