#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

#define WarpSize 32;

__device__ __forceinline__ void warpreducf32(float val) {
    #pragma unroll
    for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }

    return val;
}