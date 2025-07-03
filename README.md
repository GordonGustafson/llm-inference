About This Repo
===============

This repo contains code to run inference on the GPT2 family of LLMs.
It uses a custom CUDA kernel for the attention layer, and PyTorch for everything else.
HuggingFace is only used to load the model weights.


Case Study: Optimizing Flash Attention V1 with CUDA
===================================================

Production deployments of LLMs leverage techniques like KV-caching and reduced precision datatypes to accelerate inference.
To keep things simple and build a solid foundation in CUDA optimization, we'll skip these techniques and use GPT2 inference with Flash Attention V1 in float32 as a case study.

Our baseline will be a naïve implementation of attention using float32 in PyTorch.
We will report the runtime of generating 512 tokens of text using GPT2-small starting from the text `"Hello, I'm a language model,"`.
We will improve the runtime of this baseline by a few percentage points on an Nvidia 1660 Super GPU.

Code for each CUDA kernel can be found in the `torch_extensions` folder.

Baseline: Naïve PyTorch Implementation: 14.2 Seconds
----------------------------------------------------

The attention implementation in PyTorch boils down to this:
```
    attention_scores = queries @ keys.transpose(2, 3)
    attention_scores.masked_fill_(causal_mask.bool()[:context_length, :context_length], -torch.inf)
    attention_weights = torch.softmax(attention_scores / (head_dim ** 0.5), dim=3)

    context_vectors = attention_weights @ values
```


Version 1: Naïve Flash Attention V1 - 28.9 seconds
--------------------------------------------------

This implementation is based on [Flash Attention V1](https://arxiv.org/abs/2205.14135), with a few optimizations related to causal masking:

- Return early when the entire S block we're looking at is masked out by causal masking.
- Don't update O once we've crossed into the causally masked out region while iterating horizontally.

```
blockDim: (32, 16)
Shared Memory tile size (B_c, B_r): (48, 48)
```

(Due to a bug B_c and B_r were set to 48 when they should be set to 64 to make maximum usage of the 64Kb of shared memory on the 1660 Super GPU.)


Version 2: Avoid Bank Conflicts Accessing K - 19.1 seconds
----------------------------------------------------------

Bank conflicts occur when multiple threads simultaneously access data in the same shared memory bank, leaving one memory bank busy while the others sit idle.
Since K is stored untransposed with a row size of `d_head=64`, we hit this worst case since each thread reads the leftmost column of K, then the second to leftmost, etc.
We fix this by adding 4 bytes of padding to each row of K, causing all these accesses to the same column to hit different memory banks.
This gives us a huge speedup over the previous time of 28.9 seconds.


Version 3: Tune Tile Size and ThreadBlock Size - 18.0 seconds
-------------------------------------------------------------

```
blockDim: (32, 32)
Shared Memory tile size (B_c, B_r): (32, 32)
```

Increasing the number of total threads increases throughput, and aligning the shared memory tile size to the threadblock size reduces uncoalesced accesses to shared memory.


Version 4: Pass `--use_fast_math` flag to `nvcc` - 16.8 seconds
---------------------------------------------------------------

The `--use_fast_math` flag causes `expf()` to be replaced by the device intrinsic `__expf()`, which is much faster without a noticeable difference in inference accuracy.


Version 5: Use Warp Shuffles For Softmax Accumulation - 15.9 seconds
--------------------------------------------------------------------

Previously we used global memory for coordinating the calculations of the online softmax statistics (the maximum value and the sum of the exponentiated values with the max subtracted).
Instead we can use intra-warp communication, known as warp shuffles, and save on the high-latency accesses to global memory.
For each S block the online softmax statistics are accumulated onto the first thread in the warp, which then broadcasts to the rest of the threads in the warp.


Version 6: Miscellaneous Small Optimizations - 15.4 seconds
-----------------------------------------------------------

To keep things more concise, we apply three optimization at once in this version.

The first optimization is to templatize the `d_model`, `d_head`, `B_c`, and `B_r` arguments, which enable compiler optimizations like unrolling the `Q @ K` loop.

The second optimization is moving the division by the new softmax denominator out of the `P @ V` loop.
This optimization is present in the original Flash Attention V1 paper.

The third optimization is adding the `__restrict__` qualifier to the device pointer arguments to the kernel, which enables them to be loaded with the `LDG` PTX instruction instead of the `LD` instruction.
On some architectures this enables the load to go through a different cache that only supports read-only data.


Version 7: Use Vectorized Loads and Stores - 15.4 seconds
-------------------------------------------------------

This update uses `float4` loads and stores when possible, which can improve performance by performing fewer load and store instructions since each instruction handles 128 bits instead of 32.
Unfortunately the runtime improvement is negligible (very small decrease from the previous version).


Version 8: Update `O` in Shared Memory - 14.3 seconds
-----------------------------------------------------

While the Flash Attention V1 paper opts to maximize the shared memory dedicated to Q, K, V, and S by performing all O updates directly in global memory, I obtained better results by allocating some shared memory to O.
Computing O in shared memory and only writing it back to global memory at the very end of the kernel lets us avoid the longer latencies of global memory.


Version 9: Register Tiling Over Columns - 14.2 seconds
------------------------------------------------------

While shared memory has lower latency than global memory, registers have even lower latency than shared memory.
This version computes 2 column values of S and O per thread instead of just 1 (thread coarsening), using the value we loaded into a register for the row twice instead of once.
Doing more computation on the values we load from memory increases our arithmetic intensity, which moves us closer to being compute-bound instead of memory-bound.

We don't obtain a speedup in this version, but it sets up our next version.

```
blockDim: (16, 32)
Shared Memory tile size (B_c, B_r): (32, 32)
Register tile size (X, Y): (2, 1)
```


Version 10: Register Tiling Over Rows - 13.8 seconds
----------------------------------------------------

In this version each thread computes 4 values of S and O, 2 per column and 2 per row.
This further increases our arithmetic intensity, making better use of memory bandwidth.
To enable this we increase the shared memory tile size in the y-direction from 32 to 64.

This implementation beats our naïve PyTorch baseline of 14.2 seconds!

```
blockDim: (16, 32)
Shared Memory tile size (B_c, B_r): (32, 64)
Register tile size (X, Y): (2, 2)
```


Version 11: Support Non-Contiguous Tensors - 13.6 seconds
---------------------------------------------------------

Until now, all the kernels have required that the query, key, and value tensors be contiguous in GPU memory.
While using vectorized loads requires that the tensors' last dimension have stride 1, the kernel can easily accommodate a variable stride argument for the first dimension.
This doesn't improve the kernel runtime, but it lets us skip the `.contiguous()` calls in the Python code, resuling in a small speedup.


Version 12: Move O to Registers, Increase Tile Size - 13.3 seconds
------------------------------------------------------------------

Since each thread is responsible for computing its own values of O, we can store them in registers to save some shared memory.
We can save a bit more shared memory by alleviating K's bank conflicts with swizzling instead of padding K.
With swizzling, instead of each thread computing a dot product of Q and K starting at the first element and iterating to the last, the second thread will start at the second element and iterate forward (looping around), the third thread will start at the third element, etc.
Each thread will still compute the full dot product, but each warp will distribute its loads across different shared memory banks

Together, these two optimization save enough shared memory to fit a 64x64 shared memory tile, which lets us use a 4x4 register tile, which doubles our arithmetic intensity!

```
blockDim: (16, 16)
Shared Memory tile size (B_c, B_r): (64, 64)
Register tile size (X, Y): (4, 4)

Q shm: B_r * d_head * sizeof(float) = 64 * 64 * 4 bytes = 16 KB
K shm: B_c * d_head * sizeof(float) = 64 * 64 * 4 bytes = 16 KB
V shm: B_c * d_head * sizeof(float) = 64 * 64 * 4 bytes = 16 KB
S shm: B_r * B_c    * sizeof(float) = 64 * 64 * 4 bytes = 16 KB
total shm: 64 KB
```


Third Party Implementation: Efficient Attention - 13.2 seconds
--------------------------------------------------------------

A professional implementation achieves an even better runtime, indicating that there is still more room for further optimization!


Visual Comparison
-----------------

![Plot of inference times by number of tokens](/images/inference-times-plot.png)

Takeaways
---------

Addressing bank conflicts gave the biggest performance improvement with only a simple code change, so be sure to address them first.

Vectorized loads with `float4` made the code notably more complex, but delivered very little in terms of performance gains.
In my next CUDA kernel I'll leave vectorized loads and stores as the last optimization.


Future Work
-----------

**Memory Profiling**: Examining the peak memory usage would give a more complete picture of the advantages our kernel has over the naïve PyTorch attention kernel.

**Take advantage of reduced-precision optimizations**.

**Test other GPUs and batch sizes**.


Reproducing
-----------

```
git clone https://github.com/GordonGustafson/llm-inference.git
cd llm-inference
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
for i in $(seq 12); do
    pip uninstall -y causal_multihead_self_attention_version_${i};
    pip install --no-build-isolation torch_extensions/causal_multihead_self_attention_version_${i};
done


for attention_backend in NAIVE_PYTORCH CUSTOM_CUDA_VERSION_1 CUSTOM_CUDA_VERSION_2 CUSTOM_CUDA_VERSION_3 CUSTOM_CUDA_VERSION_4 CUSTOM_CUDA_VERSION_5 CUSTOM_CUDA_VERSION_6 CUSTOM_CUDA_VERSION_7 CUSTOM_CUDA_VERSION_8 CUSTOM_CUDA_VERSION_9 CUSTOM_CUDA_VERSION_10 CUSTOM_CUDA_VERSION_11 CUSTOM_CUDA_VERSION_12 PYTORCH_SDPA_EFFICIENT_ATTENTION; do
    python3 gpt2.py $attention_backend;
done
```

