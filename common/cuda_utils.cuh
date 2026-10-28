#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>


#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())


inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline double gflops(int M, int N, int K, float ms) {
    return 2.0 * static_cast<double>(M) * N * K / (ms * 1e-3) / 1e9;
}

inline void fill_random(std::vector<float>& v, float lo = -1.0f,
                        float hi = 1.0f, unsigned seed = 1234) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& x : v) {
        x = dist(rng);
    }
}

// Parse optional GEMM dimensions from the command line.
//
//   ./kernel              -> 1024 x 1024 x 1024
//   ./kernel 4096         -> 4096 x 4096 x 4096
//   ./kernel 4096 2048 512 -> M=4096, N=2048, K=512
inline void parse_gemm_dims(int argc, char** argv, int& M, int& N, int& K,
                            int default_size = 1024) {
    M = N = K = default_size;
    if (argc == 1) {
        return;
    }
    if (argc == 2) {
        M = N = K = std::atoi(argv[1]);
    } else if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else {
        std::fprintf(stderr, "usage: %s [size | M N K]\n", argv[0]);
        std::exit(EXIT_FAILURE);
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        std::fprintf(stderr, "M, N, K must be positive (got %d, %d, %d)\n", M,
                     N, K);
        std::exit(EXIT_FAILURE);
    }
}

inline void require_divisible(int M, int N, int K, int bm, int bn, int bk) {
    if (M % bm != 0 || N % bn != 0 || K % bk != 0) {
        std::fprintf(stderr,
                     "M, N, K must be multiples of %d, %d, %d (got %d, %d, %d)\n",
                     bm, bn, bk, M, N, K);
        std::exit(EXIT_FAILURE);
    }
}

template <typename LaunchFn>
float benchmark_ms(LaunchFn&& launch, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(iters);
}

inline int verify_gemm_sample(const std::vector<float>& a,
                              const std::vector<float>& b,
                              const std::vector<float>& c, int M, int N,
                              int K, int samples = 32, float tol = 1e-3f) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> row_dist(0, M - 1);
    std::uniform_int_distribution<int> col_dist(0, N - 1);

    int bad = 0;
    for (int s = 0; s < samples; ++s) {
        const int row = row_dist(rng);
        const int col = col_dist(rng);

        float ref = 0.0f;
        for (int i = 0; i < K; ++i) {
            ref += a[row * K + i] * b[i * N + col];
        }

        const float got = c[row * N + col];
        const float diff = std::fabs(ref - got);
        const float scale = std::max(1.0f, std::fabs(ref));
        if (diff > tol * scale) {
            if (bad < 5) {
                std::fprintf(stderr,
                             "  mismatch at (%d,%d): ref=%f got=%f diff=%f\n",
                             row, col, ref, got, diff);
            }
            ++bad;
        }
    }
    return bad;
}
