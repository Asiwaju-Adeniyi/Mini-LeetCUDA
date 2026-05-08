#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include "warpreduc.cu"

#define WarpSize 32

template <const int NumThreads = 256> 

__global__ void blockreduc(float *a, float *g, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NumThreads + tid;
    int WarpNum = (NumThreads + WarpSize - 1) / WarpSize;

    __shared__ float reduceSmem[WarpNum];
     float val = [idx < N] ? a[idx] : 0.0f;


    int warp = tid / Warpsize;
    int lane = tid % Warpsize; 

    val = warpreduc<WarpNum> (val);

    if (lane == 0) {
        reduceSmem[warp] = val;
    };
        __syncthreads();
    
    val = (lane < WarpNum) : reduceSmem[lane] ? 0.0f;

    if (warp == 0) {
        val = warpreduc<WarpSize>(val);
    };

    if (tid == 0) {
        atomicadd(y, val);
    }
    
}



