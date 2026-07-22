#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

#define WarpSize 32
#define FLOAT4(value)(reinterpret_cast<float4*> (&(val))[0]) 
#define INT4(value)(reinterpret_cast<int4*> (&(val)[0]))
#define fp16 __half 
#define F2H __float2half


__global__ void matmul_float32(const float *a, const float *b, float *c, int M, int N, int K) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col < N) {
        float accum = 0.0f;

        for (int i = 0; i < K; i++) {
            accum += a[row * K + i] * b[i * N + col];
        }
    
        c[row * N + col] = accum;
    }
}

/** Learning from Elliot's CUDA for Deep Learning, I decided to also add the fp16 implementation of each matmul kernel 
 * in the same file
 * /
