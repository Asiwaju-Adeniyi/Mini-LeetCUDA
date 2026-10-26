#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "../common/reductions.cuh"

// Batched GEMV: one thread per output element.
__global__ void gemv_kernel(const float* __restrict__ A,
                            const float* __restrict__ x, float* __restrict__ y,
                            int batch, int M, int N) {
    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch && row < M) {
        const float* batchA = A + batch_idx * M * N;
        const float* batchx = x + batch_idx * N;

        float accum = 0.0f;
        for (int col = 0; col < N; ++col) {
            accum += batchA[row * N + col] * batchx[col];
        }

        y[batch_idx * M + row] = accum;
    }
}

// Single GEMV with one warp per row.
__global__ void matVec(const float* __restrict__ a,
                       const float* __restrict__ b, float* __restrict__ c,
                       int M, int N) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= M) {
        return;
    }

    float partial = 0.0f;
    for (int i = lane; i < N; i += 32) {
        partial += a[row * N + i] * b[i];
    }

    const float sum = warpreduc<32>(partial);
    if (lane == 0) {
        c[row] = sum;
    }
}

static float cpu_dot(const float* a, const float* b, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(sum);
}

int main() {
    constexpr int batch = 8;
    constexpr int M = 1024;
    constexpr int N = 1024;

    std::vector<float> h_A(batch * M * N);
    std::vector<float> h_x(batch * N);
    std::vector<float> h_y(batch * M);
    fill_random(h_A);
    fill_random(h_x);

    float* d_A = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, h_y.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- batched gemv_kernel ----------------------------------------------
    const dim3 block(1, 32);
    const dim3 grid(1, ceil_div(M, 32), batch);

    gemv_kernel<<<grid, block>>>(d_A, d_x, d_y, batch, M, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> batch_dist(0, batch - 1);
    std::uniform_int_distribution<int> row_dist(0, M - 1);
    int bad = 0;
    for (int s = 0; s < 16; ++s) {
        const int b = batch_dist(rng);
        const int r = row_dist(rng);
        const float ref = cpu_dot(&h_A[b * M * N + r * N], &h_x[b * N], N);
        if (std::fabs(ref - h_y[b * M + r]) > 1e-3f * std::max(1.0f, std::fabs(ref))) {
            ++bad;
        }
    }
    std::printf("gemv_kernel mismatches: %d\n", bad);

    const float ms_batched = benchmark_ms([&]() {
        gemv_kernel<<<grid, block>>>(d_A, d_x, d_y, batch, M, N);
    });
    std::printf("gemv_kernel: %.3f ms\n", ms_batched);

    // ---- single matVec ----------------------------------------------------
    constexpr int WarpsPerBlock = 8;
    const dim3 block_vec(WarpsPerBlock * 32);
    const dim3 grid_vec(ceil_div(M, WarpsPerBlock));

    matVec<<<grid_vec, block_vec>>>(d_A, d_x, d_y, M, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bad = 0;
    for (int s = 0; s < 16; ++s) {
        const int r = row_dist(rng);
        const float ref = cpu_dot(&h_A[r * N], h_x.data(), N);
        if (std::fabs(ref - h_y[r]) > 1e-3f * std::max(1.0f, std::fabs(ref))) {
            ++bad;
        }
    }
    std::printf("matVec mismatches: %d\n", bad);

    const float ms_matvec = benchmark_ms([&]() {
        matVec<<<grid_vec, block_vec>>>(d_A, d_x, d_y, M, N);
    });
    std::printf("matVec: %.3f ms\n", ms_matvec);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
