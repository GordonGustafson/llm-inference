from gelu import GELU

import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # The book uses a hardcoded constant of 4.
        self.first_layer = nn.Linear(config["emb_dim"], 4 * config["emb_dim"], bias=True)
        self.second_layer = nn.Linear(4 * config["emb_dim"], config["emb_dim"], bias=True)
        self.layers = nn.Sequential(
            self.first_layer,
            GELU(),
            self.second_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)