#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

__global__ topk(const float* input, float *values, float *indices, int batch_size, int n, int k) {
    
}