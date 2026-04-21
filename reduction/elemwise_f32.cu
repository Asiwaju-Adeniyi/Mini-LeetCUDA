#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>


__global__ void naiveReduc(const float *a, const float *b, float *c, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N)

    c[idx] = a[idx] + b[idx];
}

int main() {

int N = 16 * 16;

std::vector<float> h_a(N);
std::vector<float> h_b(N);
std::vector<float> h_c(N);

for (int i = 0; i < N; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
};

float *d_a = nullptr;
float *d_b = nullptr;
float *d_c = nullptr;

size_t size = N * sizeof(float);

    int tpB = 256;
    int bpG = (N + tpB - 1) / tpB;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    naiveReduc<<<bpG, tpB>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    std::cout << gpuDuration << "ms." << std::endl;

    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


return 0;
}

