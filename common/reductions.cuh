#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// Warp-wide sum.  Every lane receives the result.
template <const int kW = 32>
__device__ __forceinline__ float warpreduc(float val) {
#pragma unroll
    for (int mask = kW >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp-wide max.  Every lane receives the result.
template <const int kW = 32>
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int mask = kW >> 1; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Block-wide sum.  Assumes blockDim.x == NumThreads and NumThreads % 32 == 0.
template <const int NumThreads = 256>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int NumWarps = (NumThreads + 31) / 32;
    __shared__ float shared[NumWarps];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    val = warpreduc<32>(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float block_val = (lane < NumWarps) ? shared[lane] : 0.0f;
        block_val = warpreduc<NumWarps>(block_val);
        if (lane == 0) {
            shared[0] = block_val;
        }
    }
    __syncthreads();

    return shared[0];
}

// Block-wide max.  Assumes blockDim.x == NumThreads and NumThreads % 32 == 0.
template <const int NumThreads = 256>
__device__ __forceinline__ float block_reduce_max(float val) {
    constexpr int NumWarps = (NumThreads + 31) / 32;
    __shared__ float shared[NumWarps];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    val = warp_reduce_max<32>(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float block_val = (lane < NumWarps) ? shared[lane] : -FLT_MAX;
        block_val = warp_reduce_max<NumWarps>(block_val);
        if (lane == 0) {
            shared[0] = block_val;
        }
    }
    __syncthreads();

    return shared[0];
}
