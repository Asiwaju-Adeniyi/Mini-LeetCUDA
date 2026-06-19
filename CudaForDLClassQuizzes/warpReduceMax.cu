#include <cstdio> 
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath> 
#include <iostream>

__device__ __forceinline__ float warpReduceMax(float value) {
    int warpSize = 32;
    float temp = -INFINITY;
    #pragma unroll 
    for (int delta = warpSize >> 1; delta > 0; delta >>= 1) {
    value = __shfl_down_sync(0xffffffff, value, delta);
    if (value > temp) {
        temp = value;
    }
    }
    return temp;
}

__device__ __forceinline__ float warpReduceMax(float value) {
    int warpSize = 32;
    float temp = 0.0f;
    #pragma unroll 
    for (int delta = warpSize >> 1; delta > 0; delta >>= 1) {
    if (value > __shfl_down_sync(0xffffffff, value, delta)) {
        temp = value;
    }
    }
    return temp;
}