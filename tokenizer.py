import tiktoken
import torch

from abc import ABC, abstractmethod
class Tokenizer(ABC):
    @abstractmethod
    def start_token(self) -> int: ...

    @abstractmethod
    def end_token(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...

    @abstractmethod
    def encode_to_torch_tensor(self, text: str, device: torch.device) -> torch.Tensor: ...


class GPT2Tokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.encoder = tiktoken.encoding_for_model("gpt-2")

    def start_token(self) -> int:
        return self.encoder.encode("<|startoftext|>")[0]

    def end_token(self) -> int:
        return self.encoder.encode("<|endoftext|>", disallowed_special=())[0]

    def encode(self, text: str) -> list:
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)

    def encode_to_torch_tensor(self, text: str, device: torch.device) -> torch.Tensor:
        tokens = self.encoder.encode(text)
        # Add in a batch dimension of 1 for now.
        return torch.tensor(tokens, device=device).unsqueeze(0)


