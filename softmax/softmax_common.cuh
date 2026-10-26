#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

#include "../common/reductions.cuh"

// Online softmax state: running max M and running denominator D.
struct __align__(8) MD {
    float M;
    float D;
};

// Combine two online-softmax states.
__device__ __forceinline__ MD combine_md(MD a, MD b) {
    const float new_m = fmaxf(a.M, b.M);
    MD out;
    out.M = new_m;
    out.D = a.D * __expf(a.M - new_m) + b.D * __expf(b.M - new_m);
    return out;
}

// Warp-wide online-softmax reduction.  Every lane receives the result.
__device__ __forceinline__ MD warp_reduce_md(MD val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        MD other;
        other.M = __shfl_xor_sync(0xffffffff, val.M, mask);
        other.D = __shfl_xor_sync(0xffffffff, val.D, mask);
        val = combine_md(val, other);
    }
    return val;
}

// Block-wide online-softmax reduction.  Assumes blockDim.x == NumThreads.
template <const int NumThreads = 256>
__device__ __forceinline__ MD block_reduce_md(MD val) {
    constexpr int NumWarps = (NumThreads + 31) / 32;
    __shared__ MD shared[NumWarps];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    val = warp_reduce_md(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        MD block_val = (lane < NumWarps) ? shared[lane] : MD{-FLT_MAX, 0.0f};
        block_val = warp_reduce_md(block_val);
        if (lane == 0) {
            shared[0] = block_val;
        }
    }
    __syncthreads();

    return shared[0];
}
