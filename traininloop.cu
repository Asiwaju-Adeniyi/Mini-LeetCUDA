#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
            __FILE__, __LINE__, cudaGetErrorString(error));   \
        cudaDeviceReset();                                    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)


