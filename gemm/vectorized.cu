#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 

#define CEIL_DIV(M, N) (((M) + (N)-1)/(N))
#define half __half
#define hadd __hadd
#define hmul __hmul
#define f2h __float2half
#define FLOAT4(val) reinterpret_cast<float4*>(&(value)[0])


