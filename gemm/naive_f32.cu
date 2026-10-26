#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

__device__ __forceinline__ void naive_gemm_impl(const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                float* __restrict__ c, int M,
                                                int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float accum = 0.0f;
        for (int i = 0; i < K; ++i) {
            accum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = accum;
    }
}

__global__ void matmul_float32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, int M, int N, int K) {
    naive_gemm_impl(a, b, c, M, N, K);
}

__global__ void sgemm_naive(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c, int M, int N, int K) {
    naive_gemm_impl(a, b, c, M, N, K);
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c(M * N);
    fill_random(h_a);
    fill_random(h_b);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));

    CUDA_CHECK(cudaMemset(d_c, 0, h_c.size() * sizeof(float)));
    matmul_float32<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const int bad_matmul =
        verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-3f);
    std::printf("matmul_float32 mismatches: %d\n", bad_matmul);

    const float matmul_ms = benchmark_ms([&]() {
        matmul_float32<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("matmul_float32: %.3f ms, %.2f GFLOP/s\n", matmul_ms,
                gflops(M, N, K, matmul_ms));

    CUDA_CHECK(cudaMemset(d_c, 0, h_c.size() * sizeof(float)));
    sgemm_naive<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const int bad_sgemm =
        verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-3f);
    std::printf("sgemm_naive mismatches: %d\n", bad_sgemm);

    const float sgemm_ms = benchmark_ms([&]() {
        sgemm_naive<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("sgemm_naive: %.3f ms, %.2f GFLOP/s\n", sgemm_ms,
                gflops(M, N, K, sgemm_ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
