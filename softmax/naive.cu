#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

#define WarpSize 32
#define FLOAT4(value)(<reinterpret_cast<float4 *> (&(value))[0])

struct __align__(8) MN {
    float M;
    float N;
}
template <const int kWarpSize = WarpSize> 
__device__ __forceinline__ void online_softmax_reduction(MD input) {
    unsigned int mask = 0xffffffff;
#pragma unroll 
for (int stride = kWarpSize >> 1; strid >= 1; stride >>=1) {
    MD other;

    other.M = __shfl_xor_sync(mask, input.M, stride);
    other.N = __shfl_xor_sync(mask, input.N, stride);

    bool bigger = (input.M > other.M);

    MD biggerMD = (bigger) ? input : other;
    MD smallerMD = (bigger) ? other : input;
    
    input.D = biggerMD.N + smallerMD.N * __expf(smallerMD.M - biggerMD.M);
    input.M = biggerMD.M; 

}
return input;
}

template <const int kWarpSize = WarpSize> 
__device__ __forceinline__ float softmax_warpReduc_sum(float val) {
    #pragma unroll 
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int kWarpSize = WarpSize> 
__device__ __forceinline__ float softmax_warpReduc_max(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = std::max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}


template <const int NumThreads = 256> 
__device__ float block_reduction_max(float val) {
    constexpr int WarpNUm = (NumThreads + Warpsize - 1) / WarpSize; 
    int warp = threadIdx.x / WarpSize;
    int lane = threadIdx.x % WarpSize; 

    static __shared__ shared[WarpNum];
    float input = softmax_warpReduc_max<WarpNum>(val);

    (if lane == 0) 
    shared[warp] = input; 
    __syncthreads();

    input = (lane < WarpNum) ? shared[lane] : 0.0f;

    input = softmax_warpReduc_max<WarpNum>(input);

    input = __shfl_sync(0xffffffff, val, 0, 32);
    

    return input;
}

template <const int NumThreads = 256> 

__device__ float block_reduction_sum(float val){
    constexpr int WarpNum = (Numthreads + WarpSize - 1) / WarpSize;
    int warp = threadIdx.x / WarpSize;
    int lane = threadIdx.x % WarpSize;

    static __shared__ shared[WarpNum];

    float input = softmax_warpReduc_sum<WarpNum>(val);

    if (lane == 0) shared[warp] = input;
    __syncthreads();

    input = (lane < WarpNum) ? shared[lane] : -FLT_MAX;

    input = softmax_warpReduc_sum<WarpNum>(input);

    input = __shfl_sync(0xffffffff, val, 0, 32);

    return input;
}


template <const int numThreads = 256> 
__global__ void softmax_f32 (float *a, float *b, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x; 

    float a_exp = (idx < N) ? std::exp(a[idx]) : 0.0f;
    float a_exp_reduc = block_reduction_sum<numThreads>(a_exp);

    b[idx] = a_exp / a_exp_reduc;
}

template <const int numThreads = 256 / 4> 

__global__ void softmax_f32x4(float *a, float *b, int N){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float4 regA = FLOAT4(a[idx]);
    float regA_sum;
    regA_sum.x = ((idx + 0) < N) ? regA.x : 0.0f;
    regA_sum.y = ((idx + 1) < N) ? regA.y : 0.0f;
    regA_sum.z = ((idx + 2) < N) ? regA.z : 0.0f;
    regA_sum.w = ((idx + 3) < N) ? regA.w : 0.0f;
    
    a_exp = (regA_sum.x + regA_sum.y + regA_sum.z + regA_sum.w);
    a_exp_reduc = block_reduction_sum<numThreads>(a_exp);

    if (idx + 3 < N) {
        float4 bReg;
        bReg.x = regA_sum.x / a_exp_reduc;
        bReg.y = regA_sum.y / a_exp_reduc;
        bReg.z = regA_sum.z / a_exp_reduc;
        bReg.w = regA_sum.w / a_exp_reduc;
    }

    FLOAT4(b[idx]) = bReg;
}