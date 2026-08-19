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
    static_assert(constexpr int iterNum = (Height * Width) / (Blocksize * elemNum), 
    "Height * Width must be multiples of Blocksize * elemNum");
    
    for (int i = 0; i < iterNum; ++i) {

       uint idx = (i * Blocksize + tid) * numElem; 
       uint row = idx / Width;
       uint col = idx % Width;

       const uint32_t dstPtr = dst + (row * Width + col) * sizeof(__nvbfloat16);
       const __nvbfloat16 *srcPtr = src + (row * stride + col);
    }
}