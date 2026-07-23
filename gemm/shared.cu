#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

#define CEIL_DIV(M, N) (((M) + (N)-1)/(N))
#define half __half
#define hadd __hadd
#define hmul __hmul
#define f2h __float2half

template <const int BLOCKSIZE>

__global__ void shared_sgemm(int M, int N, int K, float* A, float *B, float *C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKIZE * N + cCol * BLOCKSIZE;

    float accum = 0.0f;

    for (int bkIdx = 0; bkIdx < K ; bkIdx += BLOCKSIZE) {
        sA[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        sB[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            accum += sA[threadRow * BLOCKSIZE + dotIdx] * sB[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
      
    C[threadRow * N + threadCol] = accum;  
}

template <const int BLOCKSIZE> 

__global__ void hSharedGemm(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    const uint cRow = blockIdx.x;
    const int cCol = blockIdx.y; 

    __shared__ half shared[BLOCKSIZE * BLOCKSIZE];

    uint tRow = threadIdx.x / BLOCKSIZE;
    uint tCol = threadIdx.x % BLOCKSIZE;

    A += cRow * BLOCKSIZE * K; 
    B += cCol * BLOCKSIZE; 
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    half accum = f2h (0.0f);

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        
    }
}

