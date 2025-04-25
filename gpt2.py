from transformer_block import TransformerBlock
from tokenizer import GPT2Tokenizer

import torch
from torch import nn

class GPT2(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=config["vocab_size"],
                                             embedding_dim=config["emb_dim"])
        self.positional_embeddings = nn.Embedding(num_embeddings=config["context_length"],
                                                  embedding_dim=config["emb_dim"])
        dropout = nn.Dropout(p=config["drop_rate"])
        transformer_blocks = [TransformerBlock(config) for _ in range(config["n_layers"])]
        layer_norm = nn.LayerNorm(config["emb_dim"])
        output_layer = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        self.layers = nn.Sequential(*[dropout, *transformer_blocks, layer_norm, output_layer])

    def forward(self, tokens: torch.Tensor):
        batch_size, sequence_length = tokens.shape
        x = self.token_embeddings(tokens)
        x = x + self.positional_embeddings(torch.arange(0, sequence_length, device=tokens.device))
        return self.layers(x)


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    model = GPT2(GPT_CONFIG_124M)
    tokenizer = GPT2Tokenizer()
    input_tensor = tokenizer.encode_to_torch_tensor("Hello World", device=torch.device("cpu"))
    result = model(input_tensor)
    print(result.shape)