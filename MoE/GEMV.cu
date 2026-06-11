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