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

    const uint trpT[TM * TN] = {0.0f};
}
