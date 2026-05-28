#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

#define FLOAT4(val)(reinterpret_cast<float4 *> (&(value))[0])


template <const int BK, const int BM, const int BN, const int TM, const int TN>

__global__ void vectorized_kernel(int M, int N, int K, float *a, float *b, float*c) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x; 

    const int threadRow = threadIdx.x / (BN/TN);
    const int threadCol = threadIdx.x % (BN/TN);

    const int innerRowA = threadIdx.x / (BK/4);
    const int innerColA = threadIdx.x % (BK/4);
    const int innerRowB = threadIdx.x / (BN/4);
    const int innerRowB = threadIdx.x % (BN/4);

    __shared__ void sA[BM * BK];
    __shared__ void sB[BK * BN];

    const uint rpbT[TM * TN] = {0.0};
    const uint regA[TM] = {0.0};
    const uint regB[TN] = {0.0};

    A += cRow * BK * K;
    B += cCol * BN;
    C += cRow * BK * N + cCol * BN;
   

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        float4 accum = FLOAT4(A[innerRowA * K + innerColA * 4]);
        sA[(innerColA * 4 + 0) * BM + innerRowA] = accum.x;  
        sA[(innerColA * 4 + 1) * BM + innerRowA] = accum.y;
        sA[(innerColA * 4 + 2) * BM + innerRowA] = accum.z;
        sA[(innerColA * 4 + 3) * BM + innerRowA] = accum.w;

        FLOAT4(sB[innerRow * BN + innerCol * 4]) = FLOAT4(B[innerRow * N + innerCol * 4]);

        
        
    }

}

