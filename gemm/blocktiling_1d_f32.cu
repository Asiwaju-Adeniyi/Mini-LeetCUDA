#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

template <const int BM, const int BN, const int BK, const int TM>
__global__ void __launch_bounds__((BM / TM) * BN)
blocktiling_1d(const float* __restrict__ a, const float* __restrict__ b,
               float* __restrict__ c, int M, int N, int K) {
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert((BM / TM) * BN <= 1024, "block is too large");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    a += cRow * BM * K;
    b += cCol * BN;
    c += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    const int strideA = blockDim.x / BK;
    const int strideB = blockDim.x / BN;

    float threadResults[TM] = {0.0f};
    float regM[TM] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (int offset = 0; offset < BM; offset += strideA) {
            As[(innerRowA + offset) * BK + innerColA] =
                a[(innerRowA + offset) * K + innerColA];
        }
        for (int offset = 0; offset < BK; offset += strideB) {
            Bs[(innerRowB + offset) * BN + innerColB] =
                b[(innerRowB + offset) * N + innerColB];
        }
        __syncthreads();

        a += BK;
        b += BK * N;

#pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            const float regN = Bs[dotIdx * BN + threadCol];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                threadResults[i] += regM[i] * regN;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        c[(threadRow * TM + i) * N + threadCol] = threadResults[i];
    }
}

int main(int argc, char** argv) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 8;
    int M, N, K;
    parse_gemm_dims(argc, argv, M, N, K);
    require_divisible(M, N, K, BM, BN, BK);

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

    const dim3 block((BM / TM) * BN);
    const dim3 grid(N / BN, M / BM);

    blocktiling_1d<BM, BN, BK, TM><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const int bad = verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-3f);
    std::printf("blocktiling_1d mismatches: %d\n", bad);

    const float ms = benchmark_ms([&]() {
        blocktiling_1d<BM, BN, BK, TM><<<grid, block>>>(d_a, d_b, d_c, M, N,
                                                        K);
    });
    std::printf("blocktiling_1d: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
