#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

struct __align__(8) state{
    float sum;
    float sumSQ;
}

__device__ __forceinline__ float warpReduc(float val) {
    #pragma unroll 
    for (int offset = WarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fusedLayerNorm(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, 
    float* __restrict__ mean, const float* __restrict__ gamma, const float* __restrict__ beta, int N, int C) {

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        float eps = 1e-5f;

        if (tid < C) {return; };

        state fused{0.0f, 0.0f};
        int x = inp + tid * C;

        fused.sum = 

    }

