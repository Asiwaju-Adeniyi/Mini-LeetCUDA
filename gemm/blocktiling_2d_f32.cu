#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
blocktiling_2d(const float* __restrict__ a, const float* __restrict__ b,
               float* __restrict__ c, int M, int N, int K) {
    static_assert(BK % 4 == 0, "BK must be a multiple of 4");
    static_assert(BN % 4 == 0, "BN must be a multiple of 4");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert((BM * BN) % (TM * TN) == 0, "invalid tile sizes");
    static_assert((BM * BN) / (TM * TN) <= 1024, "block is too large");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    a += cRow * BM * K;
    b += cCol * BN;
    c += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    const int strideA = blockDim.x / (BK / 4);
    const int strideB = blockDim.x / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (int offset = 0; offset < BM; offset += strideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &a[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        for (int offset = 0; offset < BK; offset += strideB) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &b[(innerRowB + offset) * N + innerColB * 4])[0];
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();

        a += BK;
        b += BK * N;

#pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                regN[j] = Bs[dotIdx * BN + threadCol * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            c[(threadRow * TM + i) * N + threadCol * TN + j] =
                threadResults[i * TN + j];
        }
    }
}

int main() {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    static_assert(M % BM == 0, "M must be a multiple of BM");
    static_assert(N % BN == 0, "N must be a multiple of BN");
    static_assert(K % BK == 0, "K must be a multiple of BK");

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

    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(N / BN, M / BM);

    blocktiling_2d<BM, BN, BK, TM, TN>
        <<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const int bad = verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-3f);
    std::printf("blocktiling_2d mismatches: %d\n", bad);

    const float ms = benchmark_ms([&]() {
        blocktiling_2d<BM, BN, BK, TM, TN>
            <<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("blocktiling_2d: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
