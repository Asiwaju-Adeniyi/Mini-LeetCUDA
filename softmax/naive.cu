#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

#define WarpSize 32

struct __align__(8) MN {
    float M;
    float N;
}
template <const int kWarpSize = WarpSize> 
__device__ __forceinline__ void online_softmax_reduciton(MD input) {
    unsigned int mask = 0xffffffff;
#pragma unroll 
for (int stride = kWarpSize >> 1; strid >= 1; stride >>=1) {
    MD other;

    other.M = __shfl_xor_sync(mask, input.M, stride);
    other.D = __shfl_xor_sync(mask, input.D, stride);

    bool bigger = (input.M > other.M);

    MD biggerM = (bigger) ? input : other;
    MD smallerM = (bigger) ? other : input;
    
    input.D = biggerM.d + smallerM.d * __expf(smallerM.M - biggerM.M);
    input.M = biggerM.M; 

}
return input;
}

template <const int kWarpSize = WarpSize> 
__device__ __forceinline__ void softmax_warpReduc_sum(float val) {
    #pragma unroll 
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int kWarpSize = WarpSize> 
__device__ __forceinline__ void softmax_warpReduc_max(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = std::max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


template <const int NumThreads = 256> 
__device__ float block_reduction_max(float val) {
    constexpr int WarpNUm = (NumThreads + Warpsize - 1) / WarpSize; 
    int warp = threadIdx.x / WarpSize;
    int lane = threadIdx.x % WarpSize; 

    static __shared__ shared[WarpNum];
    float input = softmax_warpReduc_max<WarpNum>(val);

    (if lane == 0) 
    shared[warp] = input; 
    __syncthreads();

    input = (lane < WarpNum) ? shared[lane] : 0.0f;

    input = softmax_warpReduc_max<WarpNum>(input);

    input = __shfl_sync(0xffffffff, val, 0, 32);
    

    return input;
}


