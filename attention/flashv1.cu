#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath>

template <int Height, int Width, int Blocksize>

__device__ __forceinline__ void globalShared(uint32_t dst, __nvbfloat16 *src, int stride, int tid) {

    constexpr int elemNum = 16 / sizeof(__nvbfloat16);
    constexpr int iterNum = (Height * Width) / (Blocksize * elemNum);

}