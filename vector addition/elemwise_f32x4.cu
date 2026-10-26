#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

// Vectorized elementwise add: each thread handles four consecutive floats.
__global__ void elemwise_f32x4(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, int N) {
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx + 4 <= N) {
        const float4 va = reinterpret_cast<const float4*>(a + idx)[0];
        const float4 vb = reinterpret_cast<const float4*>(b + idx)[0];

        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        reinterpret_cast<float4*>(c + idx)[0] = vc;
    } else if (idx < N) {
        for (int i = 0; idx + i < N; ++i) {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int threads = 256;

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);
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

    const int blocks = ceil_div(N, 4 * threads);

    elemwise_f32x4<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        const float ref = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - ref) > 1e-6f * std::max(1.0f, std::fabs(ref))) {
            ++bad;
        }
    }
    std::printf("elemwise_f32x4 mismatches: %d/%d\n", bad, N);

    const float ms = benchmark_ms([&]() {
        elemwise_f32x4<<<blocks, threads>>>(d_a, d_b, d_c, N);
    });
    std::printf("elemwise_f32x4: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
