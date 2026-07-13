#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define WarpSize 32

__device__ __forceinline__ float warpReduc(float val) {
    #pragma unroll 
    for (int offset = WarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warpLayerNorm(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ gamma, 
    float* __restrict__ beta, float* __restrict__ mean, float* __restrict__ rstd, int N, int C) {
        
        int lane = threadIdx.x % WarpSize;
        int warp = threadIdx.x / WarpSize;
        int numWarps = blockDim.x / WarpSize;
        float eps = 1e-5f;

        int idx = blockIdx.x * numWarps + warp;

        const float* x = inp + idx * C;

        if (idx >= N) {return};
        float accum = 0.0f;

        for (int i = 0; i < C; i += WarpSize) {
             accum += x[i];
        }

        accum = warpReduc[accum];

        float m = __shfl_sync(0xffffffff, accum, 0) / C;

        if (lane == 0 && mean != nullptr) {
            mean[idx] = m;
        }

        accum = 0.0f;

        for (int i = 0; i < C; i += WarpSize) {
            float diff = x[i] - m;
            accum += diff * diff;
        }
        accum = warpReduc[accum];

        float s = __shfl_sync(0xffffffff, accum, 0);

        s = rsqrtf(s/c + eps);

        if (lane == 0 && rstd != nullptr) {
            rstd[idx] = s;
        }

        float* o = out + idx * C;
        for (int i = 0; i < C; i += WarpSize) {
            float a = s * (x[i] - m);
            o[i] = gamma[i] * n + beta[i];
        }

    }


