import torch
from torch import nn
from safetensors import safe_open


def assign_weight_and_bias(module: nn.Module, safetensors_object: safe_open, key: str, transpose_weight=False) -> None:
    if not hasattr(module, "weight") and not hasattr(module, "bias"):
        raise ValueError(f"{module} has neither 'weight' nor 'bias' attribute.")

    if hasattr(module, "weight"):
        weight_key = f"{key}.weight"
        loaded_weight = safetensors_object.get_tensor(weight_key)
        if transpose_weight:
            loaded_weight = loaded_weight.T
        if module.weight.shape != loaded_weight.shape:
            raise ValueError(f"Shape mismatch for {module} and '{weight_key}' when tranpose_weight={transpose_weight}. module weight: {module.weight.shape}. loaded weight (with optional transpose applied): {loaded_weight.shape}")
        module.weight = torch.nn.Parameter(loaded_weight)

    if hasattr(module, "bias") and module.bias is not None:
        bias_key = f"{key}.bias"
        loaded_bias = safetensors_object.get_tensor(bias_key)
        if module.bias.shape != loaded_bias.shape:
            raise ValueError(f"Shape mismatch for {module} and '{bias_key}'. module bias: {module.bias.shape}. loaded bias {loaded_bias.shape}")
        module.bias = torch.nn.Parameter(loaded_bias)
