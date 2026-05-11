#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include "warpreduc.cu"

#define WarpSize 32
#define HALF2(value) (reinterpret_cast<half2 *> (&value)[0])

template <const int kWarpSize = WarpSize> 

__device__ __forceinline__ half warp16(half val) {
     #pragma unroll 
     for (int i = kWarpSize >> 1; i >= 0; i >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, i));
     }

     return val;
}

template <const int kWarpSize = WarpSize> 

__device__ __forceinline__ float warp16_32(half val) {
float regA = __half2float(val);
#pragma unroll 
for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    regA = __shfl_xor_sync(0xffffffff, regA, mask);
}

return regA;
}

template <const int NumThreads = 256> 

__global__ void blockreduc_f16_f32(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NumThreads + tid;

    constexpr int NumWarps = (NumThreads + WarpSize - 1) / WarpSize;
    __shared__ float sharedreduc[NumWarps]; 

    int warp = tid / WarpSize;
    int lane = tid % WarpSize;

    half regA = (idx < N) : a[idx] ? __half2float(0.0f);

    #pragma unroll 
    for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
        regA = __halfadd(regA, __shfl_xor_sync(0xffffffff, regA, mask));
    }

    if (lane == 0) {
        sharedreduc[warp] = __half2float(regA);
    }
     __syncthreads();

    float sum = (lane < NumWarps) : sharedreduc ? 0.0f;
    

  if (warp == 0) (    #pragma unroll 
    for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
        sum = __halfadd(regA, __shfl_xor_sync(0xffffffff, regA, mask));
    })

    atomicadd(y, sum);


}