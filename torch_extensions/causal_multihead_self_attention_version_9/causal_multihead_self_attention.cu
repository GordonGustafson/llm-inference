#include <cuda_runtime.h>
#include <limits>
#include <stdio.h>
#include <cmath>
#include <algorithm>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <pybind11/pybind11.h>


#define THREADS_PER_WARP 32

int constexpr NUM_COLS_PER_THREAD = 2;
unsigned int constexpr ALL_THREADS_IN_WARP_MASK = 0xffffffffu;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

namespace causal_multihead_self_attention_version_9 {

// Taken from https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#setting-up-hybrid-python-c-registration,
// tweaked with https://stackoverflow.com/a/76669141.
PYBIND11_MODULE(causal_multihead_self_attention_version_9, m) {}

__device__ static inline float onlineSoftmaxSum(float const maxA,
                                                float const sumA,
                                                float const maxB,
                                                float const sumB) {
    if (sumA == 0.0f) {
        return sumB;
    } else if (sumB == 0.0f) {
        return sumA;
    } else if (maxA > maxB) {
        return sumB * expf(maxB - maxA) + sumA;
    } else {
        return sumB + sumA * expf(maxA - maxB);
    }
}

template <int d_head, int d_model, int B_c, int B_r>
__global__ void causal_multihead_self_attention_kernel(float const* const __restrict__ Q_HBM,  // size Nxd_model
                                                       float const* const __restrict__ K_HBM,  // size Nxd_model
                                                       float const* const __restrict__ V_HBM,  // size Nxd_model
                                                       float* const __restrict__ O_HBM,        // size Nxd_model
                                                       int const N) {
    extern __shared__ float sharedMemory[];
    int const T_c = CEIL_DIV(N, B_c);
    float const temperature = sqrt(d_head);

    int const B_r_bounds_checked_for_last_row = min(B_r, N - blockIdx.x * B_r);
    int const d_min_for_head = blockIdx.y * d_head;
    int const Q_row_length = d_head;
    int const O_row_length = d_head;
    // For alleviating shared memory bank conflicts
    int const K_row_length = d_head + 4;

    float* const Q = sharedMemory;
    float* const K = Q + B_r * Q_row_length;
    float* const V = K + B_c * K_row_length;
    float* const S = V + B_c * d_head;
    float* const O = S + B_c * B_r;
    float4* const Q_float4 = reinterpret_cast<float4*>(Q);
    float4* const K_float4 = reinterpret_cast<float4*>(K);
    float4* const V_float4 = reinterpret_cast<float4*>(V);
    float4* const S_float4 = reinterpret_cast<float4*>(S);
    float4* const O_float4 = reinterpret_cast<float4*>(O);
    float4 const* const Q_HBM_float4 = reinterpret_cast<float4 const*>(Q_HBM);
    float4 const* const K_HBM_float4 = reinterpret_cast<float4 const*>(K_HBM);
    float4 const* const V_HBM_float4 = reinterpret_cast<float4 const*>(V_HBM);
    float4* const O_HBM_float4 = reinterpret_cast<float4*>(O_HBM);

    float4 const zero_float4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Load Q, using threadIdx.x to help along the d_head dimension (for memory coalescing) and
    // threadIdx.y to help along the B_r dimension.
    for (int d_index = threadIdx.x; d_index < d_head / 4; d_index += blockDim.x) {
        for (int B_r_index = threadIdx.y; B_r_index < B_r_bounds_checked_for_last_row; B_r_index += blockDim.y) {
            int const row_index = blockIdx.x * B_r + B_r_index;
            Q_float4[B_r_index * (Q_row_length/4) + d_index] = Q_HBM_float4[row_index * (d_model / 4) + (d_min_for_head / 4) + d_index];
            O_float4[B_r_index * (O_row_length/4) + d_index] = zero_float4;
        }
    }

    float S_row_old_global_sum = 0.0f;
    float S_row_old_global_max = -INFINITY;

    // Iterate horizontally through different S blocks.
    for (int T_c_index = 0; T_c_index < T_c; T_c_index++) {
        int const top_row_absolute = B_r * blockIdx.x;
        int const bottow_row_absolute = top_row_absolute + B_r - 1;
        int const left_column_absolute = T_c_index * B_c;

        if (left_column_absolute > bottow_row_absolute) {
            // This entire block is masked out by causal masking.
            break;
        }

        int const num_cols_beyond_this_block_start = N - T_c_index * B_c;
        int const B_c_bounds_checked_for_last_column = min(B_c, num_cols_beyond_this_block_start);
        // Load K and V using threadIdx.x to help along the d_head dimension (for memory coalescing) and
        // threadIdx.y to help along the B_c dimension.
        for (int d_index = threadIdx.x; d_index < d_head / 4; d_index += blockDim.x) {
            for (int B_c_index = threadIdx.y; B_c_index < B_c_bounds_checked_for_last_column; B_c_index += blockDim.y) {
                int const row_index = T_c_index * B_c + B_c_index;
                K_float4[B_c_index * (K_row_length / 4) + d_index] = K_HBM_float4[row_index * (d_model / 4) + (d_min_for_head / 4) + d_index];
                V_float4[B_c_index * (d_head / 4) + d_index] = V_HBM_float4[row_index * (d_model / 4) + (d_min_for_head / 4) + d_index];
            }
        }

        // Make sure we're done writing Q, K, and V before we read them.
        __syncthreads();

        // Iterate vertically within the S block.
        // Since we use __syncthreads in this loop we have to make sure threads don't exit the function early.
        for (int B_r_index = threadIdx.y; B_r_index < CEIL_DIV(B_r_bounds_checked_for_last_row, blockDim.y) * blockDim.y; B_r_index += blockDim.y) {
            bool const row_in_bounds = B_r_index < B_r_bounds_checked_for_last_row;
            int const row_absolute = top_row_absolute + B_r_index;
            int const column_upper_bound_absolute = row_absolute + 1;
            int const column_upper_bound_within_tile = column_upper_bound_absolute - left_column_absolute;
            int const column_upper_bound = min(column_upper_bound_within_tile, B_c_bounds_checked_for_last_column);
            bool const start_column_in_row_unmasked = column_upper_bound > 0;
            int const left_S_val_column = NUM_COLS_PER_THREAD * threadIdx.x;
            bool const left_S_val_unmasked = left_S_val_column < column_upper_bound;
            float S_row_new_global_sum;
            float S_row_new_global_max;
            float localSum = 0.0f;
            float localMax = -INFINITY;

            // Initialize S to zero.
            float S_registers[NUM_COLS_PER_THREAD];
            #pragma unroll
            for (int S_reg_col = 0; S_reg_col < NUM_COLS_PER_THREAD; S_reg_col++) {
                S_registers[S_reg_col] = 0.0f;
            }

            if (row_in_bounds && left_S_val_unmasked) {
                // Compute S.
                #pragma unroll
                for (int d_index = 0; d_index < d_head / 4; d_index++) {
                    float4 const Q_val_float4 = Q_float4[B_r_index * (Q_row_length / 4) + d_index];

                    #pragma unroll
                    for (int S_reg_col = 0; S_reg_col < NUM_COLS_PER_THREAD; S_reg_col++) {
                        int const S_val_column = left_S_val_column + S_reg_col;
                        float4 const K_val_float4 = K_float4[S_val_column * (K_row_length / 4) + d_index];
                        S_registers[S_reg_col] += Q_val_float4.w * K_val_float4.w;
                        S_registers[S_reg_col] += Q_val_float4.x * K_val_float4.x;
                        S_registers[S_reg_col] += Q_val_float4.y * K_val_float4.y;
                        S_registers[S_reg_col] += Q_val_float4.z * K_val_float4.z;
                    }
                }

                // Write S to shared memory and compute localSum and localMax.
                #pragma unroll
                for (int S_reg_col = 0; S_reg_col < NUM_COLS_PER_THREAD; S_reg_col++) {
                    int const S_val_column = left_S_val_column + S_reg_col;
                    bool const S_val_unmasked = S_val_column < column_upper_bound;
                    if (S_val_unmasked) {
                        S_registers[S_reg_col] = S_registers[S_reg_col] / temperature;
                        S[B_r_index * B_c + S_val_column] = S_registers[S_reg_col];

                        localSum = onlineSoftmaxSum(localMax, localSum, S_registers[S_reg_col], 1.0f);
                        localMax = max(localMax, S_registers[S_reg_col]);
                    }
                }
            }

            // Gather the values for localSum and localMax on threadIdx.x == 0.
            // Skip bound checks and causal masking checks because it's not as simple as checking 1 row since the
            // warp shuffle spans 2 rows. Checking could be done as an optimization, you just need to be careful
            // not to cause things to run forever when N is odd.
            // ASSUMPTION: blockDim.x == 16
            for (int numActiveThreads = THREADS_PER_WARP / 4; numActiveThreads >= 1; numActiveThreads /= 2) {
                float const incomingSum = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum, numActiveThreads);
                float const incomingMax = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax, numActiveThreads);
                localSum = onlineSoftmaxSum(localMax, localSum, incomingMax, incomingSum);
                localMax = max(localMax, incomingMax);
            }

            // Broadcast the values for localSum and localMax from threadIdx.x == 0 to the other threads in the half-warp..
            // See previous warp shuffle comment about skipping bounds checks.
            int const source_lane = threadIdx.y % 2 == 0 ? 0 : 16;
            localSum = __shfl_sync(ALL_THREADS_IN_WARP_MASK, localSum, source_lane);
            localMax = __shfl_sync(ALL_THREADS_IN_WARP_MASK, localMax, source_lane);

            S_row_new_global_sum = onlineSoftmaxSum(localMax, localSum, S_row_old_global_max, S_row_old_global_sum);
            S_row_new_global_max = max(localMax, S_row_old_global_max);

            // Make sure we're done writing S before we read it.
            __syncthreads();

            if (row_in_bounds && start_column_in_row_unmasked) {
                // Compute P and O
                for (int d_index = threadIdx.x * NUM_COLS_PER_THREAD; d_index < d_head; d_index += blockDim.x * NUM_COLS_PER_THREAD) {
                    float O_registers[NUM_COLS_PER_THREAD];
                    // Set O vals to 0.
                    #pragma unroll
                    for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                        O_registers[O_reg_col] = 0.0f;
                    }

                    // Compute O vals in "strides" of 4 elements at a time.
                    int V_B_c_index = 0;
                    for (; V_B_c_index < (column_upper_bound / 4) * 4; V_B_c_index += 4) {
                        float4 S_val_float4 = S_float4[B_r_index * (B_c / 4) + (V_B_c_index / 4)];
                        S_val_float4.x = expf(S_val_float4.x - S_row_new_global_max);
                        S_val_float4.y = expf(S_val_float4.y - S_row_new_global_max);
                        S_val_float4.z = expf(S_val_float4.z - S_row_new_global_max);
                        S_val_float4.w = expf(S_val_float4.w - S_row_new_global_max);

                        #pragma unroll
                        for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                            O_registers[O_reg_col] += S_val_float4.x * V[(V_B_c_index + 0) * d_head + d_index + O_reg_col];
                            O_registers[O_reg_col] += S_val_float4.y * V[(V_B_c_index + 1) * d_head + d_index + O_reg_col];
                            O_registers[O_reg_col] += S_val_float4.z * V[(V_B_c_index + 2) * d_head + d_index + O_reg_col];
                            O_registers[O_reg_col] += S_val_float4.w * V[(V_B_c_index + 3) * d_head + d_index + O_reg_col];
                        }
                    }

                    // Compute O vals in a "stride" of 1 element at a time.
                    for (; V_B_c_index < column_upper_bound; V_B_c_index += 1) {
                        float S_val = S[B_r_index * B_c + V_B_c_index];
                        S_val = expf(S_val - S_row_new_global_max);

                        #pragma unroll
                        for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                            O_registers[O_reg_col] += S_val * V[V_B_c_index * d_head + d_index + O_reg_col];
                        }
                    }

                    // Compute and write O values
                    #pragma unroll
                    for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                        int const OIndexForThread = B_r_index * O_row_length + d_index + O_reg_col;
                        O[OIndexForThread] = (O[OIndexForThread] * expf(S_row_old_global_max - S_row_new_global_max) * S_row_old_global_sum + O_registers[O_reg_col]) / S_row_new_global_sum;
                    }
                }
            }

            S_row_old_global_sum = S_row_new_global_sum;
            S_row_old_global_max = S_row_new_global_max;

            // Make sure we're done reading S, Q, K, and V before we write them, and done writing O before we read it.
            __syncthreads();
        }
    }

    // Write O_HBM
    for (int d_index = threadIdx.x; d_index < d_head / 4; d_index += blockDim.x) {
        for (int B_r_index = threadIdx.y; B_r_index < B_r_bounds_checked_for_last_row; B_r_index += blockDim.y) {
            int const row_index = blockIdx.x * B_r + B_r_index;
            O_HBM_float4[row_index * (d_model / 4) + (d_min_for_head / 4) + d_index] = O_float4[B_r_index * (O_row_length/4) + d_index];
        }
    }
}


// Q, K, V, output are device pointers
void causal_multihead_self_attention(float const* const Q,  // size Nxd
                                     float const* const K,  // size Nxd
                                     float const* const V,  // size Nxd
                                     float* const output,   // size Nxd
                                     int const N,
                                     int const d_model,
                                     int const num_heads) {
    int maxSharedMemory;
    gpuErrchk(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0));

    int const d_head = d_model / num_heads;

    int constexpr B_c = 32;
    int constexpr B_r = 32;
    int const T_r = CEIL_DIV(N, B_r);

    dim3 const blocksPerGrid(T_r, num_heads);
    dim3 const threadsPerBlock(16, B_r);
    int const sharedMemoryBytes = (B_r * d_head          // Q
                                   + B_c * (d_head + 4)  // K
                                   + B_c * d_head        // V
                                   + B_r * B_c           // S
                                   + B_r * d_head)       // O
                                  * sizeof(float);
    if (d_head != 64) {
        throw std::invalid_argument("Head dimension must be 64.");
    }
    if (d_model == 768) {
        gpuErrchk(cudaFuncSetAttribute(causal_multihead_self_attention_kernel<64, 768, B_c, B_r>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory));
        causal_multihead_self_attention_kernel<64, 768, B_c, B_r><<<blocksPerGrid, threadsPerBlock, sharedMemoryBytes>>>(Q, K, V, output, N);
    } else if (d_model == 1024) {
        gpuErrchk(cudaFuncSetAttribute(causal_multihead_self_attention_kernel<64, 1024, B_c, B_r>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory));
        causal_multihead_self_attention_kernel<64, 1024, B_c, B_r><<<blocksPerGrid, threadsPerBlock, sharedMemoryBytes>>>(Q, K, V, output, N);
    } else if (d_model == 1280) {
        gpuErrchk(cudaFuncSetAttribute(causal_multihead_self_attention_kernel<64, 1280, B_c, B_r>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory));
        causal_multihead_self_attention_kernel<64, 1280, B_c, B_r><<<blocksPerGrid, threadsPerBlock, sharedMemoryBytes>>>(Q, K, V, output, N);
    } else if (d_model == 1600) {
        gpuErrchk(cudaFuncSetAttribute(causal_multihead_self_attention_kernel<64, 1600, B_c, B_r>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory));
        causal_multihead_self_attention_kernel<64, 1600, B_c, B_r><<<blocksPerGrid, threadsPerBlock, sharedMemoryBytes>>>(Q, K, V, output, N);
    } else {
        throw std::invalid_argument("Model dimension must be 768, 1024, 1280, or 1600.");
    }
    gpuErrchk(cudaPeekAtLastError());

#ifdef DEBUG
    std::cout << "T_r: " << T_r << std::endl;
    std::cout << "num_heads: " << num_heads << std::endl;
    std::cout << "B_c: " << B_c << std::endl;
    std::cout << "B_r: " << B_r << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "d_model: " << d_model << std::endl;
    std::cout << "d_head: " << d_head << std::endl;
    std::cout << "num_heads: " << num_heads << std::endl;
#endif
}

torch::Tensor causal_multihead_self_attention_torch(torch::Tensor Q,
                                                    torch::Tensor K,
                                                    torch::Tensor V,
                                                    int64_t num_heads) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    TORCH_CHECK(Q.dim() == 2, "Q must be a 2D tensor");
    TORCH_CHECK(K.dim() == 2, "K must be a 2D tensor");
    TORCH_CHECK(V.dim() == 2, "V must be a 2D tensor");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous")
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous")
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous")

    int N = Q.size(0);
    int d = Q.size(1);

    TORCH_CHECK(K.size(0) == N, "K must have the same sequence length as Q");
    TORCH_CHECK(V.size(0) == N, "V must have the same sequence length as Q");
    TORCH_CHECK(K.size(1) == d, "K must have the same feature dimension as Q");
    TORCH_CHECK(V.size(1) == d, "V must have the same feature dimension as Q");

    TORCH_CHECK(d % num_heads == 0, "Feature dimension of Q must be evenly divisible by the number of heads");

    torch::Tensor output = torch::empty({N, d}, Q.options());

    // Call the kernel launcher
    causal_multihead_self_attention(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        N, d, (int)num_heads
    );

    return output;
}

TORCH_LIBRARY(causal_multihead_self_attention_version_9, m) {
   // Note that "float" in the schema corresponds to the C++ double type
   // and the Python float type.
   m.def("causal_multihead_self_attention_torch(Tensor Q, Tensor K, Tensor V, int num_heads) -> Tensor");
 }

TORCH_LIBRARY_IMPL(causal_multihead_self_attention_version_9, CUDA, m) {
  m.impl("causal_multihead_self_attention_torch", &causal_multihead_self_attention_torch);
}

}
