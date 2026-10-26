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

#define HALF2(value) (reinterpret_cast<half2 *> (&value)[0])

__global__ void elemwise_fp16x8(half *a, half *b, half *c, int N) {
    int idx = 8 * (threadIdx.x + blockIdx.x * blockDim.x);

    half2 reg_a0 = HALF2(a[idx]);
    half2 reg_a1 = HALF2(a[idx + 2]);
    half2 reg_a2 = HALF2(a[idx + 4]);
    half2 reg_a3 = HALF2(a[idx + 6]);

    half2 reg_b0 = HALF2(b[idx]);
    half2 reg_b1 = HALF2(b[idx + 2]);
    half2 reg_b2 = HALF2(b[idx + 4]);
    half2 reg_b3 = HALF2(b[idx + 6]);

    half2 reg_c0, reg_c1, reg_c2, reg_c3;

    reg_c0.x = __hadd(reg_a0.x, reg_b0.x);
    reg_c0.y = __hadd(reg_a0.y, reg_b0.y);
    reg_c1.x = __hadd(reg_a1.x, reg_b1.x);
    reg_c1.y = __hadd(reg_a1.y, reg_b1.y);
    reg_c2.x = __hadd(reg_a2.x, reg_b2.x);
    reg_c2.y = __hadd(reg_a2.y, reg_b2.y);
    reg_c3.x = __hadd(reg_a3.x, reg_b3.x);
    reg_c3.y = __hadd(reg_a3.y, reg_b3.y);

    
   if (idx < N) {
    HALF2 (c[idx]) = reg_c0;
   }

   if ((idx + 2) < N) {
    HALF2 (c[idx + 2]) = reg_c1;
   }

   if ((idx + 4) < N) {
    HALF2 (c[idx + 4]) = reg_c2;
   }

   if ((idx + 6) < N) {
    HALF2 (c[idx + 6]) = reg_c3;
   }

};

int main() {
    int N = 256 * 256;

    std::vector<half> h_a(N);
    std::vector<half> h_b(N);
    std::vector<half> h_c(N);

    for (int i = 0; i < N; i++) {
        float val_a = (float)rand() / RAND_MAX;
        float val_b = (float)rand() / RAND_MAX;

        h_a[i] = __float2half(val_a);
        h_b[i] = __float2half(val_b);

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
    elemwise_fp16x8<<<bpG, tpB>>>(d_a, d_b, d_c, N);
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