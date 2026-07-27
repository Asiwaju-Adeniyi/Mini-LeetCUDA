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


template <const int BK, const int BM, const int BN, const int TM, const int TN>

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) 2D_blocktiled(int M, int N, int K, float *a, float *b, float *c) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x; 

    const uint outputPB = BM * BN;
    const uint tpB = outputPB / TM * TN;

    static_assert((tpB == blockDim.x), "threads per block not equal to block Dimension. Recheck!!");

    const uint threadRow = threadIdx.x / TM * TN;
    const uint threadCol = threadIdx.x % TM * TN;

    __shared__ void sA[BM * BK];
    __shared__ void sB[BK * BN];

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = tpB / BK; 

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = tpB / BN;

    A += cRow * BK * K;
    B += cCol * BN;
    C += cRow * BK * N + cCol * BN;

    const uint trpT[TM * TN] = {0.0};

    const uint regM{TM} = {0.0};
    const uint regN{TN} = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) { 

        for (int offset = 0; offset < BM; offset += strideA) {
            sA[(innerRowA + offset) * BK + innerColA] = A[(innerRowA + offset) * K + innerColA];
        } 

        for (int offset = 0; offset < BK; offset += strideB) {
            sB[(innerRowB + offset) * BN + innerColB] = B[(innerRowB + offset) * N + innerColB];
        }

        __syncthreads();

    for (int dotIdx = 0; dotIdx < BK) {
        for (int i = 0; i < TM; i++) {
            regM{i} = sA[(threadRow * TM + i) * BK + dotIdx];
        }
        for (int i = 0; i < TN; i++) {
            regN{i} = sB[dotIdx * BN + threadCol * TN + i];
        }

        for (int resIdxM = 0; resIdxM < TM; dotIdx++) {
            for (int resIdxN = 0; resIdxN < TN; dotIdx++) {

            }
            trpT[resIdxM * TN + resIdxN] = regM[resIdxM] * regN[resIdxN];
        }
    }
    __syncthreads();

    }

    for (int resIdxM = 0; resIdxM < TM; resIdxM++){
        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
         C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = trpT;
        }
    }
}

template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) Half_2D(int M, int N, int K, half* __restrict__ A, 
    half* __restrict__ B, half* __restrict__ C) {
        int cRow = blockIdx.y; 
        int cCol = blockIdx.x;

        int threadRow = threadIdx.x / BN;
        int threadCol = threadIdx.x % BN; 

        int innerRowA = threadIdx.x / BK;
        int innerColA = threadIdx.x % BK;
        int innerRowB = threadIdx.x / BN;
        int innerColB = threadIdx.x % BN;

        const int tpB = (BM * BN) / (TM * TN);

        const int strideA = tpB / BK;
        const int strideB = tpB % BK;

        __shared__ half sA[BM * BK];
        __shared__ half sB[BK * BN];

        half tResults[TM * TN]; 
        half regTM[TM];
        half regTN[TN];
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
                tResults[i * TN + j] = f2h(0.0f);
            }
        }

        A += cRow * BM * K;
        B += cCol * BN;
        C += cRow * BM * N + cCol * BN;


       for (int i = 0; i < K; i += BK) {
        
        for (int offsetA = 0; offsetA < BM; offsetA += strideA) {
            sA[(innerRowA + offsetA) * BK + innerCol] = A[(innerRowA + offset) * K + innerColA];
        }
        for (int offsetB = 0; offsetB < BK; offsetB += strideB) {
            sB[(innerRowB + offsetB) * BN + innerCol] = A[(innerRowB + offsetB) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;  

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int i = 0; i < TM; i++) {
                regTM[i] = sA[(threadRow * TM + i) * BK + dotIdx];
            }

            for (int j = 0; j < TN; j++) {
                regTN[j] = sA[dotIdx * BN + threadCol * TN + i];
            }
            
            
        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (resIdxN = 0; resIdxN < TN; resIdxN++) {
                int flatIdx = resIdxM * TN + residxN;

                tResults[flatIdx] = hadd(tResults[flatIdx], hmul(regM[resIdxM], regN[resIdxN]));
            }
        }
        }
        
        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (resIdxN = 0; resIdxN < TN; resIdxN++) {
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = tResults[resIdxM * TN + resIdxN];
            }

    }
}




