#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

template <const int BLOCKSIZE>
__global__ void __launch_bounds__(BLOCKSIZE * BLOCKSIZE)
shared_sgemm(const float* __restrict__ a, const float* __restrict__ b,
             float* __restrict__ c, int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    const int threadRow = threadIdx.x / BLOCKSIZE;
    const int threadCol = threadIdx.x % BLOCKSIZE;

    a += cRow * BLOCKSIZE * K;
    b += cCol * BLOCKSIZE;
    c += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float accum = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        sA[threadRow * BLOCKSIZE + threadCol] = a[threadRow * K + threadCol];
        sB[threadRow * BLOCKSIZE + threadCol] = b[threadRow * N + threadCol];
        __syncthreads();

        a += BLOCKSIZE;
        b += BLOCKSIZE * N;

#pragma unroll
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            accum += sA[threadRow * BLOCKSIZE + dotIdx] *
                     sB[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }

    c[threadRow * N + threadCol] = accum;
}

int main(int argc, char** argv) {
    constexpr int BLOCKSIZE = 32;
    int M, N, K;
    parse_gemm_dims(argc, argv, M, N, K);
    require_divisible(M, N, K, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);

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

    const dim3 block(BLOCKSIZE * BLOCKSIZE);
    const dim3 grid(N / BLOCKSIZE, M / BLOCKSIZE);

    shared_sgemm<BLOCKSIZE><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const int bad = verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-3f);
    std::printf("shared_sgemm mismatches: %d\n", bad);

    const float ms = benchmark_ms([&]() {
        shared_sgemm<BLOCKSIZE><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("shared_sgemm: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
