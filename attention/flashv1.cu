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

/* destination pointer definition is an integer literal because the ptx primitive cp.async for the shared memory state space requires a 32 bit addressing,
this is why __cvta_generic_to_shared(sharedPointer) first does the conversion. The 32 bit is visible in the register allocation "r"(dstptr);  and this is because of the limited size of the
shared memory. CUDA pointer size is naturally 64bits (inheriting from C++), but this is done because of memory constraints */
       const uint32_t dstPtr = dst + (row * Width + col) * sizeof(__nvbfloat16);
       const __nvbfloat16 *srcPtr = src + (row * stride + col);

       asm volatile(
           "cp.async.ca.shared.global [%0], [%1], 16;\n"
           :: "r"(dstPtr), "l"(srcPtr)
       );
    }
}