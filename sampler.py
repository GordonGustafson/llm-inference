import torch
from torch import nn

from tokenizer import Tokenizer

class GreedySampler:
    def __init__(self):
        super().__init__()

    def sample(self, model: nn.Module, prompt: str, max_tokens: int, tokenizer: Tokenizer, device: torch.device) -> str:
        with torch.no_grad():
            next_token = None
            text_tensor = tokenizer.encode_to_torch_tensor(prompt, device)
            while next_token != tokenizer.end_token() and text_tensor.shape[1] < max_tokens:
                token_logits = model(text_tensor)                  # shape: (batch, sequence, vocabulary)
                print(token_logits.shape)
                next_token_logits = token_logits[0, -1, :]  # shape: (vocabulary,)
                next_token = torch.argmax(next_token_logits, dim=0).item()
                next_token_tensor = torch.tensor([next_token], device=device).unsqueeze(0)
                text_tensor = torch.cat((text_tensor, next_token_tensor), 1)
            tokens_list_from_first_item_in_batch = text_tensor.tolist()[0]
            return tokenizer.decode(tokens_list_from_first_item_in_batch)

