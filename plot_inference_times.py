from causal_self_attention import ScaledDotProductAttentionBackend
from gpt2 import time_greedy_gpt2_inference

import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    hf_model_name = "openai-community/gpt2"
    attention_backend_values = [ScaledDotProductAttentionBackend.CUSTOM_CUDA, ScaledDotProductAttentionBackend.NAIVE_PYTORCH]
    device = torch.device("cuda")
    prompt = "Hello, I'm a language model,"
    max_tokens_values = [32, 64, 128, 256, 512]

    timing_results = {}
    for attention_backend in attention_backend_values:
        for max_tokens in max_tokens_values:
            inference_timing = time_greedy_gpt2_inference(hf_model_name=hf_model_name,
                                                          attention_backend=attention_backend,
                                                          device=device,
                                                          prompt=prompt,
                                                          max_tokens=max_tokens)
            print(f"time elapsed by first run: {inference_timing.first_run_execution_time_seconds} seconds")
            print(f"time elapsed by second run: {inference_timing.second_run_execution_time_seconds} seconds")
            print(f"generated text: {inference_timing.generated_text}")
            timing_results[(attention_backend, max_tokens)] = inference_timing.second_run_execution_time_seconds

    print(timing_results)

    for attention_backend in attention_backend_values:
        timings = [timing_results[(attention_backend, max_tokens)] for max_tokens in max_tokens_values]
        plt.plot(max_tokens_values, timings, label=f"{attention_backend.name}")

    plt.legend()
    plt.savefig("inference-times.png")
