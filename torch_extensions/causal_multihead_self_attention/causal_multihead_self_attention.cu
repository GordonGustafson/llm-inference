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
int constexpr NUM_ROWS_PER_THREAD = 2;
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

namespace causal_multihead_self_attention {

// Taken from https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#setting-up-hybrid-python-c-registration,
// tweaked with https://stackoverflow.com/a/76669141.
PYBIND11_MODULE(causal_multihead_self_attention, m) {}

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

    float S_row_old_global_sum[NUM_ROWS_PER_THREAD];
    float S_row_old_global_max[NUM_ROWS_PER_THREAD];
    #pragma unroll
    for (int row = 0; row < NUM_ROWS_PER_THREAD; row++) {
        S_row_old_global_sum[row] = 0.0f;
        S_row_old_global_max[row] = -INFINITY;
    }

    int const shm_tile_top_row_hbm = B_r * blockIdx.x;
    int const shm_tile_bottom_row_hbm = shm_tile_top_row_hbm + B_r - 1;

    // Iterate horizontally through different S blocks.
    for (int T_c_index = 0; T_c_index < T_c; T_c_index++) {
        int const shm_tile_left_column_hbm = T_c_index * B_c;
        if (shm_tile_left_column_hbm > shm_tile_bottom_row_hbm) {
            // This entire block is masked out by causal masking.
            break;
        }

        int const num_cols_beyond_this_block_start = N - shm_tile_left_column_hbm;
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
        for (int reg_tile_top_row_shm = threadIdx.y * NUM_ROWS_PER_THREAD;
             reg_tile_top_row_shm < CEIL_DIV(B_r_bounds_checked_for_last_row, blockDim.y * NUM_ROWS_PER_THREAD) * blockDim.y * NUM_ROWS_PER_THREAD;
             reg_tile_top_row_shm += blockDim.y * NUM_ROWS_PER_THREAD) {

            bool const reg_tile_top_row_shm_in_bounds = reg_tile_top_row_shm < B_r_bounds_checked_for_last_row;
            int const reg_tile_top_row_hbm = shm_tile_top_row_hbm + reg_tile_top_row_shm;
            int const reg_tile_top_row_column_causal_upper_bound_hbm = reg_tile_top_row_hbm + 1;
            int const reg_tile_top_row_column_upper_bound_shm = min(reg_tile_top_row_column_causal_upper_bound_hbm - shm_tile_left_column_hbm,
                                                                    B_c_bounds_checked_for_last_column);

            int const reg_tile_bottom_row_hbm = reg_tile_top_row_hbm + NUM_ROWS_PER_THREAD - 1;
            int const reg_tile_bottom_row_column_causal_upper_bound_hbm = reg_tile_bottom_row_hbm + 1;
            int const reg_tile_bottom_row_column_upper_bound_shm = min(reg_tile_bottom_row_column_causal_upper_bound_hbm - shm_tile_left_column_hbm,
                                                                       B_c_bounds_checked_for_last_column);

            int const reg_tile_left_column_shm = NUM_COLS_PER_THREAD * threadIdx.x;
            int const reg_tile_left_column_hbm = shm_tile_left_column_hbm + reg_tile_left_column_shm;
            bool const reg_tile_bottom_left_unmasked = reg_tile_left_column_hbm <= reg_tile_bottom_row_hbm;

            float S_row_new_global_sum[NUM_ROWS_PER_THREAD];
            float S_row_new_global_max[NUM_ROWS_PER_THREAD];
            float localSum[NUM_ROWS_PER_THREAD];
            float localMax[NUM_ROWS_PER_THREAD];

            // Initialize S and localSum to zero, and localMax to -INFINITY..
            float S_registers[NUM_ROWS_PER_THREAD][NUM_COLS_PER_THREAD];
            #pragma unroll
            for (int S_reg_row = 0; S_reg_row < NUM_ROWS_PER_THREAD; S_reg_row++) {
                localSum[S_reg_row] = 0.0f;
                localMax[S_reg_row] = -INFINITY;
                #pragma unroll
                for (int S_reg_col = 0; S_reg_col < NUM_COLS_PER_THREAD; S_reg_col++) {
                    S_registers[S_reg_row][S_reg_col] = 0.0f;
                }
            }

            if (reg_tile_top_row_shm_in_bounds && reg_tile_bottom_left_unmasked) {
                // Compute S.
                #pragma unroll
                for (int d_index = 0; d_index < d_head / 4; d_index++) {
                    float4 Q_reg_float4[NUM_ROWS_PER_THREAD];
                    #pragma unroll
                    for (int S_reg_row = 0; S_reg_row < NUM_ROWS_PER_THREAD; S_reg_row++) {
                        Q_reg_float4[S_reg_row] = Q_float4[(reg_tile_top_row_shm + S_reg_row) * (Q_row_length / 4) + d_index];
                    }

                    #pragma unroll
                    for (int S_reg_col = 0; S_reg_col < NUM_COLS_PER_THREAD; S_reg_col++) {
                        int const S_col_shm = reg_tile_left_column_shm + S_reg_col;
                        float4 const K_reg_float4 = K_float4[S_col_shm * (K_row_length / 4) + d_index];
                        #pragma unroll
                        for (int S_reg_row = 0; S_reg_row < NUM_ROWS_PER_THREAD; S_reg_row++) {
                            S_registers[S_reg_row][S_reg_col] += Q_reg_float4[S_reg_row].w * K_reg_float4.w;
                            S_registers[S_reg_row][S_reg_col] += Q_reg_float4[S_reg_row].x * K_reg_float4.x;
                            S_registers[S_reg_row][S_reg_col] += Q_reg_float4[S_reg_row].y * K_reg_float4.y;
                            S_registers[S_reg_row][S_reg_col] += Q_reg_float4[S_reg_row].z * K_reg_float4.z;
                        }
                    }
                }

                // Write S to shared memory and compute localSum and localMax.

                #pragma unroll
                for (int S_reg_row = 0; S_reg_row < NUM_ROWS_PER_THREAD; S_reg_row++) {
                    #pragma unroll
                    for (int S_reg_col = 0; S_reg_col < NUM_COLS_PER_THREAD; S_reg_col++) {
                        int const S_col_shm = reg_tile_left_column_shm + S_reg_col;
                        int const shm_row_hbm = shm_tile_top_row_hbm + reg_tile_top_row_shm + S_reg_row;
                        int const shm_row_column_causal_upper_bound_hbm = shm_row_hbm + 1;
                        // TODO: bounds check column in S_registers computation loop
                        int const shm_row_column_upper_bound_shm = min(shm_row_column_causal_upper_bound_hbm - shm_tile_left_column_hbm,
                                                                       B_c_bounds_checked_for_last_column);
                        bool const S_val_in_bounds_and_unmasked = S_col_shm < shm_row_column_upper_bound_shm;
                        if (S_val_in_bounds_and_unmasked) {
                            S_registers[S_reg_row][S_reg_col] = S_registers[S_reg_row][S_reg_col] / temperature;
                            S[(reg_tile_top_row_shm + S_reg_row) * B_c + S_col_shm] = S_registers[S_reg_row][S_reg_col];

                            localSum[S_reg_row] = onlineSoftmaxSum(localMax[S_reg_row], localSum[S_reg_row], S_registers[S_reg_row][S_reg_col], 1.0f);
                            localMax[S_reg_row] = max(localMax[S_reg_row], S_registers[S_reg_row][S_reg_col]);
                        }
                    }
                }
            }

            // Gather the values for localSum and localMax on threadIdx.x == 0.
            // Skip bound checks and causal masking checks because it's not as simple as checking 1 row since the
            // warp shuffle spans 2 rows. Checking could be done as an optimization, you just need to be careful
            // not to cause things to run forever when N is odd.
            // ASSUMPTION: blockDim.x == 16
            #pragma unroll
            for (int row = 0; row < NUM_ROWS_PER_THREAD; row++) {
                for (int numActiveThreads = THREADS_PER_WARP / 4; numActiveThreads >= 1; numActiveThreads /= 2) {
                    float const incomingSum = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localSum[row], numActiveThreads);
                    float const incomingMax = __shfl_down_sync(ALL_THREADS_IN_WARP_MASK, localMax[row], numActiveThreads);
                    localSum[row] = onlineSoftmaxSum(localMax[row], localSum[row], incomingMax, incomingSum);
                    localMax[row] = max(localMax[row], incomingMax);
                }

                // Broadcast the values for localSum and localMax from threadIdx.x == 0 to the other threads in the half-warp..
                // See previous warp shuffle comment about skipping bounds checks.
                int const source_lane = threadIdx.y % 2 == 0 ? 0 : 16;
                localSum[row] = __shfl_sync(ALL_THREADS_IN_WARP_MASK, localSum[row], source_lane);
                localMax[row] = __shfl_sync(ALL_THREADS_IN_WARP_MASK, localMax[row], source_lane);

                S_row_new_global_sum[row] = onlineSoftmaxSum(localMax[row], localSum[row], S_row_old_global_max[row], S_row_old_global_sum[row]);
                S_row_new_global_max[row] = max(localMax[row], S_row_old_global_max[row]);
            }


            // Make sure we're done writing S before we read it.
            __syncthreads();

            if (reg_tile_top_row_shm_in_bounds) {
                // Compute P and O
                for (int d_index = threadIdx.x * NUM_COLS_PER_THREAD; d_index < d_head; d_index += blockDim.x * NUM_COLS_PER_THREAD) {
                    float O_registers[NUM_ROWS_PER_THREAD][NUM_COLS_PER_THREAD];
                    // Set O vals to 0.
                    #pragma unroll
                    for (int O_reg_row = 0; O_reg_row < NUM_ROWS_PER_THREAD; O_reg_row++) {
                        #pragma unroll
                        for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                            O_registers[O_reg_row][O_reg_col] = 0.0f;
                        }
                    }

                    // Compute O vals in "strides" of 4 elements at a time.
                    // No bounds checking needed in this loop because we're only going until the top column's causal upper bound
                    // (the most restrictive upper bound).
                    int V_B_c_index = 0;
                    for (; V_B_c_index < (reg_tile_top_row_column_upper_bound_shm / 4) * 4; V_B_c_index += 4) {
                        float4 S_reg_float4[NUM_ROWS_PER_THREAD];
                        #pragma unroll
                        for (int O_reg_row = 0; O_reg_row < NUM_ROWS_PER_THREAD; O_reg_row++) {
                            S_reg_float4[O_reg_row] = S_float4[(reg_tile_top_row_shm + O_reg_row) * (B_c / 4) + (V_B_c_index / 4)];
                            S_reg_float4[O_reg_row].x = expf(S_reg_float4[O_reg_row].x - S_row_new_global_max[O_reg_row]);
                            S_reg_float4[O_reg_row].y = expf(S_reg_float4[O_reg_row].y - S_row_new_global_max[O_reg_row]);
                            S_reg_float4[O_reg_row].z = expf(S_reg_float4[O_reg_row].z - S_row_new_global_max[O_reg_row]);
                            S_reg_float4[O_reg_row].w = expf(S_reg_float4[O_reg_row].w - S_row_new_global_max[O_reg_row]);
                        }

                        #pragma unroll
                        for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                            float const V_reg_x = V[(V_B_c_index + 0) * d_head + d_index + O_reg_col];
                            float const V_reg_y = V[(V_B_c_index + 1) * d_head + d_index + O_reg_col];
                            float const V_reg_z = V[(V_B_c_index + 2) * d_head + d_index + O_reg_col];
                            float const V_reg_w = V[(V_B_c_index + 3) * d_head + d_index + O_reg_col];
                            #pragma unroll
                            for (int O_reg_row = 0; O_reg_row < NUM_ROWS_PER_THREAD; O_reg_row++) {
                                bool const reg_tile_row_in_bounds = reg_tile_top_row_shm + O_reg_row < B_r_bounds_checked_for_last_row;
                                if (reg_tile_row_in_bounds) {
                                    O_registers[O_reg_row][O_reg_col] += S_reg_float4[O_reg_row].x * V_reg_x;
                                    O_registers[O_reg_row][O_reg_col] += S_reg_float4[O_reg_row].y * V_reg_y;
                                    O_registers[O_reg_row][O_reg_col] += S_reg_float4[O_reg_row].z * V_reg_z;
                                    O_registers[O_reg_row][O_reg_col] += S_reg_float4[O_reg_row].w * V_reg_w;
                                }
                            }
                        }
                    }

                    // Compute O vals in a "stride" of 1 element at a time.
                    for (; V_B_c_index < reg_tile_bottom_row_column_upper_bound_shm; V_B_c_index += 1) {
                        float S_reg[NUM_ROWS_PER_THREAD];

                        #pragma unroll
                        for (int O_reg_row = 0; O_reg_row < NUM_ROWS_PER_THREAD; O_reg_row++) {
                            S_reg[O_reg_row] = S[(reg_tile_top_row_shm + O_reg_row) * B_c + V_B_c_index];
                            S_reg[O_reg_row] = expf(S_reg[O_reg_row] - S_row_new_global_max[O_reg_row]);
                        }

                        #pragma unroll
                        for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                            float const V_reg = V[V_B_c_index * d_head + d_index + O_reg_col];
                            #pragma unroll
                            for (int O_reg_row = 0; O_reg_row < NUM_ROWS_PER_THREAD; O_reg_row++) {
                                bool const reg_tile_row_in_bounds = reg_tile_top_row_shm + O_reg_row < B_r_bounds_checked_for_last_row;
                                if (reg_tile_row_in_bounds) {
                                    int const reg_tile_row_hbm = reg_tile_top_row_hbm + O_reg_row;
                                    int const reg_tile_column_hbm = shm_tile_left_column_hbm + V_B_c_index;
                                    if (reg_tile_column_hbm <= reg_tile_row_hbm) {
                                        O_registers[O_reg_row][O_reg_col] += S_reg[O_reg_row] * V_reg;
                                    }
                                }
                            }
                        }
                    }

                    // Compute and write O values
                    #pragma unroll
                    for (int O_reg_row = 0; O_reg_row < NUM_ROWS_PER_THREAD; O_reg_row++) {
                        #pragma unroll
                        for (int O_reg_col = 0; O_reg_col < NUM_COLS_PER_THREAD; O_reg_col++) {
                            int const OIndexForThread = (reg_tile_top_row_shm + O_reg_row) * O_row_length + d_index + O_reg_col;
                            O[OIndexForThread] = (O[OIndexForThread] * expf(S_row_old_global_max[O_reg_row] - S_row_new_global_max[O_reg_row]) * S_row_old_global_sum[O_reg_row] + O_registers[O_reg_row][O_reg_col]) / S_row_new_global_sum[O_reg_row];
                        }
                    }
                }
            }

            #pragma unroll
            for (int row = 0; row < NUM_ROWS_PER_THREAD; row++) {
                S_row_old_global_sum[row] = S_row_new_global_sum[row];
                S_row_old_global_max[row] = S_row_new_global_max[row];
            }

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
    // 64KB is available on Turing GPUs. Change this if using a GPU with a larger value.
    int maxSharedMemory = 65536;

    int const d_head = d_model / num_heads;

    int constexpr B_c = 32;
    int constexpr B_r = 64;
    int const T_r = CEIL_DIV(N, B_r);

    dim3 const blocksPerGrid(T_r, num_heads);
    dim3 const threadsPerBlock(16, 32);
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

TORCH_LIBRARY(causal_multihead_self_attention, m) {
   // Note that "float" in the schema corresponds to the C++ double type
   // and the Python float type.
   m.def("causal_multihead_self_attention_torch(Tensor Q, Tensor K, Tensor V, int num_heads) -> Tensor");
 }

TORCH_LIBRARY_IMPL(causal_multihead_self_attention, CUDA, m) {
  m.impl("causal_multihead_self_attention_torch", &causal_multihead_self_attention_torch);
}

}
