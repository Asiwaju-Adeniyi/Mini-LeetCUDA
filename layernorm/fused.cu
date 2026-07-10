#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define WarpSize 32

__device__ __forceinline__ warpReduc(float val) {
    #pragma unroll 
    for (int offset = WarpSize >> 1; offset >= 1; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fusedLayerNorm(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ gamma, 
    float* __restrict__ beta, float* __restrict__ mean, float* __restrict__ rstd, int N, int C) {
        
    }


