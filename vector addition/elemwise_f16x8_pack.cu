#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

// fp16 elementwise add with a single 128-bit load and store per thread.
__global__ void elemwise_f16x8_pack(const __half* __restrict__ a,
                                    const __half* __restrict__ b,
                                    __half* __restrict__ c, int N) {
    const int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N) {
        return;
    }

    if (idx + 8 <= N) {
        const float4 va = reinterpret_cast<const float4*>(a + idx)[0];
        const float4 vb = reinterpret_cast<const float4*>(b + idx)[0];

        const __half2* pa = reinterpret_cast<const __half2*>(&va);
        const __half2* pb = reinterpret_cast<const __half2*>(&vb);

        float4 vc;
        __half2* pc = reinterpret_cast<__half2*>(&vc);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            pc[i] = __hadd2(pa[i], pb[i]);
        }

        reinterpret_cast<float4*>(c + idx)[0] = vc;
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

    elemwise_f16x8_pack<<<blocks, threads>>>(d_a, d_b, d_c, N);
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
    std::printf("elemwise_f16x8_pack mismatches: %d/%d\n", bad, N);

    const float ms = benchmark_ms([&]() {
        elemwise_f16x8_pack<<<blocks, threads>>>(d_a, d_b, d_c, N);
    });
    std::printf("elemwise_f16x8_pack: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
