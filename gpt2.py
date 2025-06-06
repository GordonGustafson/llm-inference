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
from dataclasses import dataclass

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

def _get_gpt2_config(hf_model_name: str, attention_backend: ScaledDotProductAttentionBackend):
    COMMON_GPT_CONFIG = {
        "scaled_dot_product_attention_backend": attention_backend,
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": True,  # Query-Key-Value bias
    }
    HF_MODEL_NAME_TO_GPT_CONFIG = {
        "openai-community/gpt2":        {**COMMON_GPT_CONFIG, **{"emb_dim": 768,  "n_heads": 12, "n_layers": 12}},
        "openai-community/gpt2-medium": {**COMMON_GPT_CONFIG, **{"emb_dim": 1024, "n_heads": 16, "n_layers": 24}},
        "openai-community/gpt2-large":  {**COMMON_GPT_CONFIG, **{"emb_dim": 1280, "n_heads": 20, "n_layers": 36}},
        "openai-community/gpt2-xl":     {**COMMON_GPT_CONFIG, **{"emb_dim": 1600, "n_heads": 25, "n_layers": 48}},
    }

    return HF_MODEL_NAME_TO_GPT_CONFIG[hf_model_name]

def make_gpt_model(hf_model_name: str,
                   attention_backend: ScaledDotProductAttentionBackend,
                   device: torch.device):
    config = _get_gpt2_config(hf_model_name=hf_model_name,
                              attention_backend=attention_backend)
    model = GPT2(config)
    model.load_weights_from_huggingface(hf_model_name)
    model.eval()
    model.to(device)
    return model

@dataclass
class InferenceTiming:
    first_run_execution_time_seconds: float
    second_run_execution_time_seconds: float
    generated_text: str

def time_greedy_gpt_inference(hf_model_name: str,
                              attention_backend: ScaledDotProductAttentionBackend,
                              device: torch.device,
                              prompt: str,
                              max_tokens: int):
    model = make_gpt_model(hf_model_name=hf_model_name,
                           attention_backend=attention_backend,
                           device=device)
    tokenizer = GPT2Tokenizer()
    sampler = GreedySampler()

    print("Executing warmup...")
    num_tokens_in_prompt = len(tokenizer.encode(prompt))
    num_tokens_to_generate_for_warmup = 3
    start_time_ns = time.perf_counter_ns()
    _ = sampler.sample(model=model,
                       prompt=prompt,
                       # TODO: check that this doesn't exceed the max context length
                       max_tokens=num_tokens_in_prompt + num_tokens_to_generate_for_warmup,
                       tokenizer=tokenizer,
                       device=device)
    end_time_ns = time.perf_counter_ns()
    first_run_execution_time_seconds = (end_time_ns - start_time_ns) / (10 ** 9)

    torch.manual_seed(0)

    print("Executing timed run...")
    start_time_ns = time.perf_counter_ns()
    generated_text = sampler.sample(model=model,
                                    prompt=prompt,
                                    max_tokens=max_tokens,
                                    tokenizer=tokenizer,
                                    device=device)
    end_time_ns = time.perf_counter_ns()
    second_run_execution_time_seconds = (end_time_ns - start_time_ns) / (10 ** 9)

    return InferenceTiming(first_run_execution_time_seconds=first_run_execution_time_seconds,
                           second_run_execution_time_seconds=second_run_execution_time_seconds,
                           generated_text=generated_text)


if __name__ == "__main__":
    inference_timing = time_greedy_gpt_inference(hf_model_name="openai-community/gpt2",
                                                 attention_backend=ScaledDotProductAttentionBackend.CUSTOM_CUDA,
                                                 device=torch.device("cuda"),
                                                 prompt="Hello, I'm a language model,",
                                                 max_tokens=500)
    print(f"time elapsed by first run: {inference_timing.first_run_execution_time_seconds} seconds")
    print(f"time elapsed by second run: {inference_timing.second_run_execution_time_seconds} seconds")
    print(f"generated text: {inference_timing.generated_text}")
