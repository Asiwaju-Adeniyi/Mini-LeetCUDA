#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

template <const int BLOCKSIZE>
__global__ void __launch_bounds__(BLOCKSIZE * BLOCKSIZE)
hSharedGemm(const __half* __restrict__ a, const __half* __restrict__ b,
            __half* __restrict__ c, int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    __shared__ __half sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ __half sB[BLOCKSIZE * BLOCKSIZE];

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
            accum += __half2float(sA[threadRow * BLOCKSIZE + dotIdx]) *
                     __half2float(sB[dotIdx * BLOCKSIZE + threadCol]);
        }
        __syncthreads();
    }

    c[threadRow * N + threadCol] = __float2half(accum);
}

int main(int argc, char** argv) {
    constexpr int BLOCKSIZE = 32;
    int M, N, K;
    parse_gemm_dims(argc, argv, M, N, K);
    require_divisible(M, N, K, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);

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

    const dim3 block(BLOCKSIZE * BLOCKSIZE);
    const dim3 grid(N / BLOCKSIZE, M / BLOCKSIZE);

    hSharedGemm<BLOCKSIZE><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
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
    std::printf("hSharedGemm mismatches: %d\n", bad);

    const float ms = benchmark_ms([&]() {
        hSharedGemm<BLOCKSIZE><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    });
    std::printf("hSharedGemm: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
