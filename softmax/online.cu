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
__device__ __forceinline__ MD softmax_warp_reduc(MD input) {
    int mask = 0xffffffff;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    #pragma unroll 
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
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


template <const int NumThreads = 256>
__global__ void online_softmax_f32(const float* inp, float* out, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int WarpNum = (NumThreads + WarpSize - 1) / WarpSize;

    int warp = threadIdx.x / WarpSize;
    int lane = threadIdx.x % WarpSize;

    __shared__ MD shared[WarpNum];

    MD val;
    val.M = (idx < N) ? inp[idx] : -FLT_MAX;
    val.D = (idx < N) ? 1.0f : 0.0f;

    MD res1 = softmax_warp_reduc<WarpSize>(val);

    if (lane == 0) {
        shared[warp] = res1;
    }
    __syncthreads();

    if (threadIdx.x < WarpSize) {
        MD res2;
        res2 = (lane < WarpNum) ? shared[lane] : MD{-FLT_MAX, 0.0f};
         
        res1 = softmax_warp_reduc<WarpNum>(res2);

        if (threadIdx.x == 0)
            shared[0] = res1;
           __syncthreads();

     if (threadIdx.x == 0)
       res2 = share[0];


       float normalizer = __fdividef(1.0f, res2.D);

       output[idx] = __expf(inp[idx] - res2.M) / normalizer;

    }
}
