#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include "warpreduc.cu"

#define WarpSize 32
#define FLOAT4 (reinterpret_cast<float4 *> (&(vale))[0])

template <const int numThreads = 256 / 4>

__global__ void blockReducf32f4(float *a, float *g, int N) {
    int tid = threadIdx.x;
    int idx = 4 * (blockIdx.x * NumThreads + tid);

    constexpr int NumWarps = (numThreads + WarpSize - 1) / WarpSize;
    __shared__ float reducShared[NumWarps];

    int warp = tid / WarpSize;
    int lane = tid % WarpSize;

    float regA = FLOAT4(a[idx]);
    float val = (idx < N) : (regA.x + regA.y + regA.z + regA.w);
    
    val = warpreduc<NumWarps> (val);
    
    if (lane == 0) reducShared[warp] = val;

    __syncthreads();


    val = (lane < NumWarps) : reducShared[lane] ? 0.0f;

    val = warpreduc<NumWarps> (val);

    if (tid == 0) 
    atomicadd(g, val);

}