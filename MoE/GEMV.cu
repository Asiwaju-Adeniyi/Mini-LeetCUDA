#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

__global__ void gemv_kernel(const float *A, const float *x, float* y, int batch, int M, int N) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch && row < M) {

        float accum = 0.0f;
        const float* batchA = A + batch_idx * M * N;
        const float* batchx = x + batch_idx * N;

        accum += batchA[row * N + col] * batchx[col];
    }

    y[batch_idx * M + row] = accum;
}

__device__ __forceinline__ float warpReduceSum(float val){
    uint WarpSize = 32;
    #pragma unroll

    for (int delta = WarpSize >> 1; delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;
}

__global__ void matVec(const float *a, const float *b, float *c, int M, int N) {
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M) return;

    float partialAccum = 0.0f;
    int tid = threadIdx.x;
    for (int i = tid;  i < N; i += blockDim.x) {
    partialAccum += a[row * N + i] * b[i]; 

    float temp = warpReduce(partialAccum);

    if (tid == 0) {
       c[row] = temp;
    }
 } }