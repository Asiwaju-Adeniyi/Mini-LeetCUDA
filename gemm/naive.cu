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
#define half __half 
#define f2h __float2half


__global__ void matmul_float32(const float *a, const float *b, float *c, int M, int N, int K) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

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
 * */


 __global__ void hGemm(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        half accum = f2h(0.0f);

        for (int i = 0; i < N; ++i) {
            accum = __hadd(accum, __hmul(A[x * K + i], B[i * N + y]))
        }

        C[x * N + y] = accum;
    }
 }
