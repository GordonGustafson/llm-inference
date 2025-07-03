from causal_self_attention import ScaledDotProductAttentionBackend, ALL_VERSIONED_CUSTOM_CUDA_BACKENDS
from gpt2 import time_greedy_gpt2_inference

import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    hf_model_name = "openai-community/gpt2"
    attention_backend_values = (ALL_VERSIONED_CUSTOM_CUDA_BACKENDS
                                + [ScaledDotProductAttentionBackend.NAIVE_PYTORCH,
                                   ScaledDotProductAttentionBackend.PYTORCH_SDPA_EFFICIENT_ATTENTION])
    device = torch.device("cuda")
    prompt = "Hello, I'm a language model,"
    max_tokens_values = [32, 64, 128, 256, 512]

    timing_results = {}
    max_memory_allocated_results = {}
    for attention_backend in attention_backend_values:
        for max_tokens in max_tokens_values:
            inference_timing = time_greedy_gpt2_inference(hf_model_name=hf_model_name,
                                                          attention_backend=attention_backend,
                                                          device=device,
                                                          prompt=prompt,
                                                          max_tokens=max_tokens)
            print(f"time elapsed for generating {max_tokens} tokens with {attention_backend}: {inference_timing.second_run_execution_time_seconds} seconds")
            print(f"generated text: {inference_timing.generated_text}")
            timing_results[(attention_backend, max_tokens)] = inference_timing.second_run_execution_time_seconds
            max_memory_allocated_results[(attention_backend, max_tokens)] = inference_timing.max_gpu_memory_allocated_bytes

    print(timing_results)
    print(max_memory_allocated_results)

    for attention_backend in attention_backend_values:
        timings = [timing_results[(attention_backend, max_tokens)] for max_tokens in max_tokens_values]
        plt.plot(max_tokens_values, timings, label=f"{attention_backend.name}")

    plt.legend()
    plt.title("GPT-small Inference Timings")
    plt.xlabel("Num Tokens Generated")
    plt.ylabel("Time Elapsed (Seconds)")
    plt.savefig("inference-times.png")
    plt.clf()

    for attention_backend in [ScaledDotProductAttentionBackend.NAIVE_PYTORCH,
                              ScaledDotProductAttentionBackend.PYTORCH_SDPA_EFFICIENT_ATTENTION,
                              ScaledDotProductAttentionBackend.CUSTOM_CUDA_VERSION_12]:
        memory_used = [max_memory_allocated_results[(attention_backend, max_tokens)] for max_tokens in max_tokens_values]
        plt.plot(max_tokens_values, memory_used, label=f"{attention_backend.name}")

    plt.legend()
    plt.title("GPT-small Memory Usage")
    plt.xlabel("Num Tokens Generated")
    plt.ylabel("Peak Memory Usage (Bytes)")
    plt.savefig("memory-usage.png")
