from gelu import GELU

import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # The book uses a hardcoded constant of 4.
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"], bias=True),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"], bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)