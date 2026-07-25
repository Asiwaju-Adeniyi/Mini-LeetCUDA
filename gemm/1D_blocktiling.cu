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

template <const int BK, const int BM, const int BN, const int TM>

__global__ void 1D_blocktiled(int M, int N, int K, float *a, float*b, float* c) {
    
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    static_assert((BM * BK) == blockIdx.x, "You sure?");
    static_assert((BN * BK) == blockIdx.x, "You damn sure?");
    const uint innerRowA = threadRow / BK;
    const uint innerColA = threadCol % BK;
    const uint innerRowB = threadRow / BN;
    const uint innerColB = threadCol % BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;



    float accum[TM] = {0.0};

        for (int bkIdx = 0; bkIdx < K; bkIdx++) {
            sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
            sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

            __syncthreads();

            A += BK;
            B += BN * N;

            for (int id = 0; id < BK; id++) {
                float accumB = sB[innerRowB * BN + innerColB];

                for (int dot = 0; dot < BK; dot++) {
                   accum[dot] += sA[(threadRow * TM + dot) * K + id] * accumB;
                } 
            }
             __syncthreads();
        }

        for (int resIdx = 0; resIdx < TM; resIdx++) {
            C[(threadRow * TM + resIdx) * N + threadCol] = accum[resIdx];
        }
}

template <const int BK, const int BM, const int BN, const int TM>

__global__ void half_1D_blocktiled(int M, int N, int K, half* __restrict a, half* __restrict__ b, half* __restrict__ c) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    const uint inRowA = threadIdx.x / BK;
    const uint inColA = threadIdx.x % BK;
    const uint inRowB = threadIdx.x / BN;
    const uint inColB = threadIdx.x % BN;
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    __shared__ half sA[BM * BK];
    __shared__ half sB[BK * BN];

    half tResults[TM] = {half(0.0f)};

    for (int i = 0; i < K; i++) {
        sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        sB[innerRowB * BN + innerColB] = B[innerRowA * N + innerColB];

        __syncthreads();

        A += BK;
        B += cCol * BN; 
        C += cRow * BM * N + cCol * BN;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            half accumB = sB[dotIdx * BN + threadCol];

            for (int resIdx = 0; resIdx < TM; resIdx++) {
                tResults[resIdx] = hadd(tResults[resIdx], hmul(sA[(threadRow * TM + resIdx) * BK + dotIdx], accumB));
            }
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; resIdx++) {
        C[[threadRow * TM + resIdx] * N + threadCol] = tResults[resIdx];
    }


}
