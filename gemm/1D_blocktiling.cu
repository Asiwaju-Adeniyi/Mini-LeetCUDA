#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

template <const int BK, const int BM, const int BN, const int TM>

__global__ void 1D_blocktiled(int M, int N, int K, float *a, float*b, float* c) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    __shared__ void sA[BM * BK];
    __shared__ void sB[BK * BN];

    static_assert((BM * BK) == blockIdx.x, "You sure?");
    static_assert((BN * BK) == blockIdx.x, "You damn sure?");

    A += cRow * BK * K;
    B += cCol * BN;
    C += cRow * BK * K + cCol * BN;

    float accum = {0.0;

        for (int bkIdx = 0; bkIdx < K)
}