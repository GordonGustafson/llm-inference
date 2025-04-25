import tiktoken
import torch

class GPT2Tokenizer:
    def __init__(self):
        super().__init__()
        self.encoder = tiktoken.encoding_for_model("gpt-2")

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)

    def encode_to_torch_tensor(self, text: str, device: torch.device) -> torch.Tensor:
        tokens = self.encoder.encode(text)
        # Add in a batch dimension of 1 for now.
        return torch.tensor(tokens, device=device).unsqueeze(0)


