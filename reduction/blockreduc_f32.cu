#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "warpreduc.cuh"

#define WarpSize 32

// Block reduction: each block reduces NumThreads values, then atomically adds
// its partial sum into a single global scalar.
template <const int NumThreads = 256>
__global__ void blockreduc(const float* __restrict__ a, float* __restrict__ g,
                           int N) {
    static_assert(NumThreads % WarpSize == 0, "NumThreads must be a multiple of 32");

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * NumThreads + tid;
    constexpr int warpNum = (NumThreads + WarpSize - 1) / WarpSize;

    __shared__ float reduceSmem[warpNum];

    float val = (idx < N) ? a[idx] : 0.0f;

    const int warp = tid / WarpSize;
    const int lane = tid % WarpSize;

    val = warpreduc<WarpSize>(val);

    if (lane == 0) {
        reduceSmem[warp] = val;
    }
    __syncthreads();

    val = (lane < warpNum) ? reduceSmem[lane] : 0.0f;

    if (warp == 0) {
        val = warpreduc<warpNum>(val);
    }

    if (tid == 0) {
        atomicAdd(g, val);
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int NumThreads = 256;

    std::vector<float> h_a(N);
    fill_random(h_a);

    float* d_a = nullptr;
    float* d_g = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(NumThreads);
    const dim3 grid(ceil_div(N, NumThreads));

    CUDA_CHECK(cudaMemset(d_g, 0, sizeof(float)));
    blockreduc<NumThreads><<<grid, block>>>(d_a, d_g, N);
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
    std::printf("blockreduc: ref=%.6f got=%.6f diff=%.6f\n", ref, h_g, diff);
    if (diff > 1e-2 * scale) {
        std::printf("blockreduc: MISMATCH\n");
    } else {
        std::printf("blockreduc: MATCH\n");
    }

    const float ms = benchmark_ms([&]() {
        CUDA_CHECK(cudaMemsetAsync(d_g, 0, sizeof(float)));
        blockreduc<NumThreads><<<grid, block>>>(d_a, d_g, N);
    });
    std::printf("blockreduc: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_g));
    return 0;
}
