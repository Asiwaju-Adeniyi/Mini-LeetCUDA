#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "warpreduc.cuh"

#define WarpSize 32

// fp16 input, fp32 accumulation.
template <const int NumThreads = 256>
__global__ void blockreduc_f16_f32(const __half* __restrict__ a,
                                   float* __restrict__ y, int N) {
    static_assert(NumThreads % WarpSize == 0,
                  "NumThreads must be a multiple of 32");

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * NumThreads + tid;
    constexpr int NumWarps = (NumThreads + WarpSize - 1) / WarpSize;

    __shared__ float sharedreduc[NumWarps];

    const int warp = tid / WarpSize;
    const int lane = tid % WarpSize;

    float val = (idx < N) ? __half2float(a[idx]) : 0.0f;

    val = warpreduc<WarpSize>(val);

    if (lane == 0) {
        sharedreduc[warp] = val;
    }
    __syncthreads();

    val = (lane < NumWarps) ? sharedreduc[lane] : 0.0f;

    if (warp == 0) {
        val = warpreduc<NumWarps>(val);
    }

    if (tid == 0) {
        atomicAdd(y, val);
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int NumThreads = 256;

    std::vector<float> h_a_f32(N);
    std::vector<__half> h_a(N);
    fill_random(h_a_f32);
    for (int i = 0; i < N; ++i) {
        h_a[i] = __float2half(h_a_f32[i]);
    }

    __half* d_a = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));

    const dim3 block(NumThreads);
    const dim3 grid(ceil_div(N, NumThreads));

    CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
    blockreduc_f16_f32<NumThreads><<<grid, block>>>(d_a, d_y, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_y = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    double ref = 0.0;
    for (const __half x : h_a) {
        ref += __half2float(x);
    }
    const double diff = std::fabs(ref - h_y);
    const double scale = std::max(1.0, std::fabs(ref));
    std::printf("blockreduc_f16_f32: ref=%.6f got=%.6f diff=%.6f\n", ref, h_y,
                diff);
    if (diff > 1e-2 * scale) {
        std::printf("blockreduc_f16_f32: MISMATCH\n");
    } else {
        std::printf("blockreduc_f16_f32: MATCH\n");
    }

    const float ms = benchmark_ms([&]() {
        CUDA_CHECK(cudaMemsetAsync(d_y, 0, sizeof(float)));
        blockreduc_f16_f32<NumThreads><<<grid, block>>>(d_a, d_y, N);
    });
    std::printf("blockreduc_f16_f32: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
