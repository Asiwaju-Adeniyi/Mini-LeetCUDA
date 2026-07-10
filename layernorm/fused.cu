#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define WarpSize 32

__device__ __forceinline__ warpReduc(float val) {
    #pragma unroll 
    for (int offset = WarpSize >> 1; offset >= 1; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


