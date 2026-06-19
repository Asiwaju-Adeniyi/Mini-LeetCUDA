#include <cstdio> 
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath> 
#include <iostream>

__device__ __forceinline__ float warpReduceMax(float value) {
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        float temp = __shfl_down_sync(0xffffffff, value, delta);
        value = fmaxf(value, temp);
    }
    return value;
}

__global__ void testWarpMax(const float *in, float *out) {
    int tid = threadIdx.x;
    float v = in[tid];
    float result = warpReduceMax(v);
    if (tid == 0) {
        out[0] = result;
    }
}

int main() {
    const int N = 32;
    std::vector<float> h_A(N);
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_out;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int tpB = 32;
    int bpG = 1;
    testWarpMax<<<bpG, tpB>>>(d_A, d_out);

    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // verify against CPU
    float cpu_max = h_A[0];
    for (int i = 1; i < N; i++) cpu_max = fmaxf(cpu_max, h_A[i]);

    printf("GPU max: %f\n", h_out);
    printf("CPU max: %f\n", cpu_max);
    printf("%s\n", (h_out == cpu_max) ? "MATCH" : "MISMATCH");

    cudaFree(d_A);
    cudaFree(d_out);
    return 0;
}
