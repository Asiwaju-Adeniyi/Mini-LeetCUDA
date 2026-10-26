#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "warpreduc.cuh"

#define WarpSize 32

// Vectorized block reduction: each thread reduces four floats.
template <const int numThreads = 256 / 4>
__global__ void blockReducf32f4(const float* __restrict__ a,
                                float* __restrict__ g, int N) {
    static_assert(numThreads % WarpSize == 0,
                  "numThreads must be a multiple of 32");

    const int tid = threadIdx.x;
    const int idx = 4 * (blockIdx.x * numThreads + tid);

    constexpr int NumWarps = (numThreads + WarpSize - 1) / WarpSize;
    __shared__ float reducShared[NumWarps];

    const int warp = tid / WarpSize;
    const int lane = tid % WarpSize;

    float val = 0.0f;
    if (idx + 3 < N) {
        const float4 regA = reinterpret_cast<const float4*>(&a[idx])[0];
        val = regA.x + regA.y + regA.z + regA.w;
    } else {
        for (int i = 0; i < 4 && idx + i < N; ++i) {
            val += a[idx + i];
        }
    }

    val = warpreduc<WarpSize>(val);

    if (lane == 0) {
        reducShared[warp] = val;
    }
    __syncthreads();

    val = (lane < NumWarps) ? reducShared[lane] : 0.0f;

    if (warp == 0) {
        val = warpreduc<NumWarps>(val);
    }

    if (tid == 0) {
        atomicAdd(g, val);
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int numThreads = 256 / 4;

    std::vector<float> h_a(N);
    fill_random(h_a);

    float* d_a = nullptr;
    float* d_g = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(numThreads);
    const dim3 grid(ceil_div(N, 4 * numThreads));

    CUDA_CHECK(cudaMemset(d_g, 0, sizeof(float)));
    blockReducf32f4<numThreads><<<grid, block>>>(d_a, d_g, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_g = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_g, d_g, sizeof(float), cudaMemcpyDeviceToHost));

    double ref = 0.0;
    for (float x : h_a) {
        ref += x;
    }
    const double diff = std::fabs(ref - h_g);
    const double scale = std::max(1.0, std::fabs(ref));
    std::printf("blockReducf32f4: ref=%.6f got=%.6f diff=%.6f\n", ref, h_g,
                diff);
    if (diff > 1e-2 * scale) {
        std::printf("blockReducf32f4: MISMATCH\n");
    } else {
        std::printf("blockReducf32f4: MATCH\n");
    }

    const float ms = benchmark_ms([&]() {
        CUDA_CHECK(cudaMemsetAsync(d_g, 0, sizeof(float)));
        blockReducf32f4<numThreads><<<grid, block>>>(d_a, d_g, N);
    });
    std::printf("blockReducf32f4: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_g));
    return 0;
}
