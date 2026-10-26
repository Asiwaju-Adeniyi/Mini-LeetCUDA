#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

// Naive fp16 GEMM with fp32 accumulation.
// C[M,N] = A[M,K] * B[K,N], all matrices row-major.
__global__ void hGemm(const __half* __restrict__ a,
                      const __half* __restrict__ b, __half* __restrict__ c,
                      int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float accum = 0.0f;
        for (int i = 0; i < K; ++i) {
            accum += __half2float(a[row * K + i]) *
                     __half2float(b[i * N + col]);
        }
        c[row * N + col] = __float2half(accum);
    }
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    std::vector<float> h_a_f32(M * K);
    std::vector<float> h_b_f32(K * N);
    fill_random(h_a_f32);
    fill_random(h_b_f32);

    std::vector<__half> h_a(M * K);
    std::vector<__half> h_b(K * N);
    std::vector<__half> h_c(M * N);
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = __float2half(h_a_f32[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_b[i] = __float2half(h_b_f32[i]);
    }

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    __half* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));

    hGemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    std::vector<float> h_c_f32(M * N);
    for (int i = 0; i < M * N; ++i) {
        h_c_f32[i] = __half2float(h_c[i]);
    }

    const int bad = verify_gemm_sample(h_a_f32, h_b_f32, h_c_f32, M, N, K, 32,
                                       1e-2f);
    std::printf("hGemm mismatches: %d\n", bad);

    const float ms = benchmark_ms([&]() {
        hGemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("hGemm: %.3f ms, %.2f GFLOP/s\n", ms, gflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
