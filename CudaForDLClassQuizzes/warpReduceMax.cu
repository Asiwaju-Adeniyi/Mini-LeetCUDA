#include <cstdio> 
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath> 
#include <iostream>

__device__ __forceinline__ float warpReduceMax(float value) {
int warpSize = 32;
float temp = 0.0f;
#pragma unroll 
for (int delta = warpSize >> 1; delta > 0; delta >>= 1) {
    temp = __shfl_down_sync(0xffffffff, value, delta);
    value = fmaxf(value, temp);
  };
  return value;
}
