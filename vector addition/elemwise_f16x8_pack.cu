#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <iostream>

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

__global__ void ldst128_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (threadIdx.x + blockIdx.x * blockDim.x);
    
    half regA[8], regB[8], regC[8];
LDST128BITS(regA[0]) = LDST128BITS(a[idx]);
LDST128BITS(regB[0]) = LDST128BITS(b[idx]);

    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
       HALF2(regC[i]) = __hadd2(HALF2(regA[i]), HALF2(regB[i]));
    }

  
        if ((idx + 7) < N) {
            LDST128BITS(c[idx]) = LDST128BITS(regC[0]);
        } else {
            for (int i = 0; idx + i < N; i++) {
                c[idx + i] = __hadd(a[idx + i], b[idx + i]); 
            }
        }
    }


    int main() {
    int N = 256 * 256;

    std::vector<half> h_a(N);
    std::vector<half> h_b(N);
    std::vector<half> h_c(N);

    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half(rand() / (float)RAND_MAX);
        h_b[i] = __float2half(rand() / (float)RAND_MAX);

    }

    half *d_a = nullptr;
    half *d_b = nullptr;
    half *d_c = nullptr;

    size_t size = N * sizeof(half);

    int tpB = 256/8;
    int bpG = (N + 256 - 1) / 256;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ldst128_kernel<<<bpG, tpB>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuDuration = 0.0f;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    std::cout << gpuDuration << " ms" << std::endl;

        cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);



   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

   return 0;


}
