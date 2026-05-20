#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 
#include "naive.cu"


#define WarpSize 32
#define FLOAT4(value)(reinterpret_cast<float4*> (&(val))[0]) 
#define INT4(value)(reinterpret_cast<int4*> (&(val)[0]))

template <const int BM = 32, const int BN = 32, const int BK = 32>

__global__ void gemm_f32_tiled(float *a, float *b, float *c, float *M, float *N, float *K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x * tx;

    __shared__ s_a[32][32]; 
    __shared__ s_b[32][32];


}