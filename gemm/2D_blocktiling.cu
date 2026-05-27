#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

template <const int BK, const int BM, const int BN, const int TM, TN>

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) 2D_blocktiled(int M, int N, int K, float *a, float *b, float *c) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x; 

    const uint outputPB = BM * BN;
    const uint tpB = outputPB / TM * TN;

    static_assert((tpB == blockDim.x), "You're a disappointment.");

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

        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            regM{i} = sA[(threadRow * TM + i) * BK + dotIdx];
        }
        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
            regN{i} = sB[(i * TN + dotIdx) * BN + threadCol];
        }

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            trpT[dotIdx] = regM[threadRow * TM + dotIdx] * regN[dotIdx * TN + threadCol];
        }

        __syncthreads();

    }

    for (int resIdxM = 0; resIdxM < TM; resIdxM++){
        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
         C[(innerRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = trpT;
        }
    }
}
