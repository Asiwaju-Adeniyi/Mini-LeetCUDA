#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

// fp16 elementwise add, eight halves per thread using four half2 operations.
__global__ void elemwise_f16x8(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               __half* __restrict__ c, int N) {
    const int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N) {
        return;
    }

    if (idx + 8 <= N) {
        const __half2 a0 = reinterpret_cast<const __half2*>(a + idx)[0];
        const __half2 a1 = reinterpret_cast<const __half2*>(a + idx)[1];
        const __half2 a2 = reinterpret_cast<const __half2*>(a + idx)[2];
        const __half2 a3 = reinterpret_cast<const __half2*>(a + idx)[3];

        const __half2 b0 = reinterpret_cast<const __half2*>(b + idx)[0];
        const __half2 b1 = reinterpret_cast<const __half2*>(b + idx)[1];
        const __half2 b2 = reinterpret_cast<const __half2*>(b + idx)[2];
        const __half2 b3 = reinterpret_cast<const __half2*>(b + idx)[3];

        reinterpret_cast<__half2*>(c + idx)[0] = __hadd2(a0, b0);
        reinterpret_cast<__half2*>(c + idx)[1] = __hadd2(a1, b1);
        reinterpret_cast<__half2*>(c + idx)[2] = __hadd2(a2, b2);
        reinterpret_cast<__half2*>(c + idx)[3] = __hadd2(a3, b3);
    } else {
        for (int i = 0; idx + i < N; ++i) {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int threads = 256;

    std::vector<float> h_a_f32(N);
    std::vector<float> h_b_f32(N);
    fill_random(h_a_f32);
    fill_random(h_b_f32);

    std::vector<__half> h_a(N);
    std::vector<__half> h_b(N);
    std::vector<__half> h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = __float2half(h_a_f32[i]);
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

    const int blocks = ceil_div(N, 8 * threads);

    elemwise_f16x8<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        const float ref = __half2float(h_a[i]) + __half2float(h_b[i]);
        if (std::fabs(__half2float(h_c[i]) - ref) > 5e-3f) {
            ++bad;
        }
    }
    std::printf("elemwise_f16x8 mismatches: %d/%d\n", bad, N);

    const float ms = benchmark_ms([&]() {
        elemwise_f16x8<<<blocks, threads>>>(d_a, d_b, d_c, N);
    });
    std::printf("elemwise_f16x8: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
