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

__global__ void gemV_address_tracking(const float *a, const float *b, float *c, int M, int N) {
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M) {
        float accum = 0.0f;
        for (int i = 0; i < N; i++) {
            if (row < 3 && i == 0) {
                printf("thread %u: a address = %p, b address = %p\n",
                       row, (void*)&a[row * N + i], (void*)&b[i]);
            }
            accum += a[row * N + i] * b[i];
        }
        c[row] = accum;
    }
}