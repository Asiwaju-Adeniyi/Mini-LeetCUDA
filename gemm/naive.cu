#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

#define WarpSize 32
#define FLOAT4(value)(reinterpret_cast<float4*> (&(val))[0]) 


__global__ void matmul_float32(const float *a, const float *b, float *c, int m, int n, int k)