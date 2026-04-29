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
__device__ __forceinline__ void softmax_sum(float val) {
    #pragma unroll 
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int kWarpSize = WarpSize> 
__device__ __forceinline__ void softmax_max(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = std::max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


