#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
blocktiling_2d_f16(const __half* __restrict__ a,
                   const __half* __restrict__ b, __half* __restrict__ c,
                   int M, int N, int K) {
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert((BM * BN) % (TM * TN) == 0, "invalid tile sizes");
    static_assert((BM * BN) / (TM * TN) <= 1024, "block is too large");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    a += cRow * BM * K;
    b += cCol * BN;
    c += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    const int strideA = blockDim.x / BK;
    const int strideB = blockDim.x / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

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
                regM[i] =
                    __half2float(As[(threadRow * TM + i) * BK + dotIdx]);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                regN[j] =
                    __half2float(Bs[dotIdx * BN + threadCol * TN + j]);
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
                __float2half(threadResults[i * TN + j]);
        }
    }
}

int main(int argc, char** argv) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    int M, N, K;
    parse_gemm_dims(argc, argv, M, N, K);
    require_divisible(M, N, K, BM, BN, BK);

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

    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(N / BN, M / BM);

    blocktiling_2d_f16<BM, BN, BK, TM, TN>
        <<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    std::vector<float> h_c_f32(M * N);
    for (int i = 0; i < M * N; ++i) {
        h_c_f32[i] = __half2float(h_c[i]);
    }

    const int bad =
        verify_gemm_sample(h_a_f32, h_b_f32, h_c_f32, M, N, K, 32, 1e-2f);
    std::printf("blocktiling_2d_f16 mismatches: %d\n", bad);

    const float ms = benchmark_ms([&]() {
        blocktiling_2d_f16<BM, BN, BK, TM, TN>
            <<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("blocktiling_2d_f16: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
