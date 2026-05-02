#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <cmath> 
#include "naive.cu"

#define WarpSize 32
#define FLOAT4(value)(<reinterpret_cast<float4 *> (&(value))[0])

template <const int NumThreads = 256> 
__global__ void safe_softmax_f32(const float *a, float *b, int N) {
    const int idx = threadIdx.x + blockIdx.x + blockDim.x;

    float val = (idx < N) ? a[idx] : -FLT_MAX;
    float max_val = block_reduction_max<NumThreads>(val);

    float regA = (idx < N) ? __expf(val[idx] - max_val) : 0.0f;
    float regA_exp = block_reduction_sum<NumThreads>(regA);

    b[idx] = regA / regA_exp;
}

template <const int NumThreads = 256/4> 
__global__ void safe_softmax_f32x4(const float *a, float *b, int N){
    int idx = 4 * (threadIdx.x + blockDim.x * blockIdx.x);

    float4 regA = FLOAT4(a[idx]);

    regA.x = ((idx + 0) < N) ? regA.x : -FLT_MAX;
    regA.y = ((idx + 1) < N) ? regA.y : -FLT_MAX;
    regA.z = ((idx + 2) < N) ? regA.z : -FLT_MAX;
    regA.w = ((idx + 3) < N) ? regA.w : -FLT_MAX;

    float val = regA.x;
    val = fmaxf(val, regA.y);
    val = fmaxf(val, regA.z);
    val = fmaxf(val, regA.w);

    float max_val = block_reduction_max<NumThreads>(val);

    float4 valReg; 
    valReg.x = ((idx + 0) < N) ? expf(regA.x - max_val) : 0.0f;
    valReg.y = ((idx + 1) < N) ? expf(regA.y - max_val) : 0.0f;
    valReg.z = ((idx + 2) < N) ? expf(regA.y - max_val) : 0.0f;
    valReg.w = ((idx + 3) < N) ? expf(regA.z - max_val) : 0.0f;

    float valSum = valSum.x + valSum.y + valSum.z + valSum.w;
    float valSum_exp = block_reduction_sum<NumThreads>(valSum);

    

    if ((idx + 3) < N) {
        float4 reg_y;
        reg_y.x = valSum.x / valSum_exp;
        reg_y.y = valSum.y / valSum_exp;
        reg_y.z = valSum.z / valSum_exp;
        reg_y.w = valSum.w / valSum_exp;

        FLOAT4(b[idx]) = reg_y;
    }
    
}