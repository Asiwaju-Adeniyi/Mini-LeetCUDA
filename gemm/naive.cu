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