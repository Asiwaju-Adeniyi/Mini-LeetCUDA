#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>

#define FLOAT4(value)(reinterpret_cast<float4 *> (&(value)) [0])

__global__ void elemwise_f32x4(const float* a, const float* b, float* c, int N) {
    int idx = 4 * (threadIdx.x + blockDim.x * blockIdx.x);

    if (idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;

        FLOAT4(c[idx]) = reg_c;
    }
}

int main() {
    int N = 256 * 256;

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    for (int i = 0; i < N; i++) {
        h_a = rand() / (float)RAND_MAX;
        h_b = rand() / (float)RAND_MAX;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    size_t size = N * sizeof(float);

    int tpB = 256/4;
    int bpG = (N / 256 - 1) / 256;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    elemwise_f32x4<<<bpG, tpB>>>(d_a, d_b, d_c, N);
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

