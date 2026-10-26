#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

__global__ void vectorized_gemm(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (row < M && col < N) {
        float4 accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < K; ++i) {
            const float av = a[row * K + i];
            const float4 bv =
                reinterpret_cast<const float4*>(&b[i * N + col])[0];
            accum.x += av * bv.x;
            accum.y += av * bv.y;
            accum.z += av * bv.z;
            accum.w += av * bv.w;
        }
        reinterpret_cast<float4*>(&c[row * N + col])[0] = accum;
    }
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    static_assert(N % 4 == 0, "N must be a multiple of 4");

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
    const dim3 grid(ceil_div(N / 4, block.x), ceil_div(M, block.y));

    vectorized_gemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const int bad = verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-3f);
    std::printf("vectorized_gemm mismatches: %d\n", bad);

    const float ms = benchmark_ms([&]() {
        vectorized_gemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("vectorized_gemm: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
