#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <cmath> 
#include "naive.cu"

#define WarpSize 32
#define FLOAT4(value)(<reinterpret_cast<float4 *> (&(value))[0])


__global__ void online_softmax_f32(const float* inp, float* out, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    
}
