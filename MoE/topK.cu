#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

__global__ topk(const float* input, float *values, float *indices, int batch_size, int n, int k) {
    int batch_idx = blockIdx.x;

    if (batch_idx < batch_size) {
        const float* input_row = input + batch_idx * n;
        float* v_row = values + batch_idx * k;
        float* i_row = indices + batch_idx * k;
    }
}