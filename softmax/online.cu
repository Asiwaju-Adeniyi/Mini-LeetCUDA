#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <cmath> 
#include "naive.cu"

#define WarpSize 32
#define FLOAT4(value)(<reinterpret_cast<float4 *> (&(value))[0])

struct __align__(8) MD {
    float M;
    float D;
}
template < const int kWarpSize = WarpSize> 
__device__ __forceinline__ MD online_softmax(MD input) {
    int mask = 0xffffffff;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    #pragma unroll 
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        other.M = __shfl_xor_sync(mask, input.M, stride);
        other.D = __shfl_xor_sync(mask, input.D, stride);

        bool bigger = (input.M > other.M);

        MD biggerM = (bigger) ? input : other;
        MD smallerM = (bigger) ? other : input;

        input.D = biggerM.D + smallerM.D * __expf(smallerM.M - biggerM.M);
        input.M = biggerM.M;
    }
    return input;
    
}



__global__ void online_softmax_f32(const float* inp, float* out, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;


    
}
