#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "softmax_common.cuh"

// Naive row-wise softmax: exp(x) / sum(exp(x)).
// One block per row.
template <const int NumThreads = 256>
__global__ void softmax_f32(const float* __restrict__ input,
                            float* __restrict__ output, int rows, int cols) {
    const int row = blockIdx.x;
    const float* x = input + row * cols;
    float* y = output + row * cols;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += __expf(x[i]);
    }

    const float sum = block_reduce_sum<NumThreads>(local_sum);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = __expf(x[i]) / sum;
    }
}

// Vectorized version: each thread handles four consecutive columns.
template <const int NumThreads = 256 / 4>
__global__ void softmax_f32x4(const float* __restrict__ input,
                              float* __restrict__ output, int rows, int cols) {
    const int row = blockIdx.x;
    const float* x = input + row * cols;
    float* y = output + row * cols;

    float local_sum = 0.0f;
    for (int i = threadIdx.x * 4; i < cols; i += blockDim.x * 4) {
        const float4 v = reinterpret_cast<const float4*>(&x[i])[0];
        local_sum += __expf(v.x) + __expf(v.y) + __expf(v.z) + __expf(v.w);
    }

    const float sum = block_reduce_sum<NumThreads>(local_sum);

    for (int i = threadIdx.x * 4; i < cols; i += blockDim.x * 4) {
        const float4 v = reinterpret_cast<const float4*>(&x[i])[0];
        float4 o;
        o.x = __expf(v.x) / sum;
        o.y = __expf(v.y) / sum;
        o.z = __expf(v.z) / sum;
        o.w = __expf(v.w) / sum;
        reinterpret_cast<float4*>(&y[i])[0] = o;
    }
}

static std::vector<float> cpu_naive_softmax(const std::vector<float>& input,
                                            int rows, int cols) {
    std::vector<float> output(input.size());
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum += std::exp(static_cast<double>(input[r * cols + c]));
        }
        for (int c = 0; c < cols; ++c) {
            output[r * cols + c] = static_cast<float>(
                std::exp(static_cast<double>(input[r * cols + c])) / sum);
        }
    }
    return output;
}

static float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

int main() {
    constexpr int rows = 1024;
    constexpr int cols = 1024;
    static_assert(cols % 4 == 0, "cols must be a multiple of 4");

    std::vector<float> h_in(rows * cols);
    std::vector<float> h_out(rows * cols);
    fill_random(h_in, -1.0f, 1.0f);
    const std::vector<float> h_ref = cpu_naive_softmax(h_in, rows, cols);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, h_out.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- softmax_f32 -------------------------------------------------------
    softmax_f32<256><<<rows, 256>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::printf("softmax_f32 max abs diff: %.6g\n",
                max_abs_diff(h_out, h_ref));

    const float ms_f32 = benchmark_ms([&]() {
        softmax_f32<256><<<rows, 256>>>(d_in, d_out, rows, cols);
    });
    std::printf("softmax_f32: %.3f ms\n", ms_f32);

    // ---- softmax_f32x4 -----------------------------------------------------
    softmax_f32x4<64><<<rows, 64>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::printf("softmax_f32x4 max abs diff: %.6g\n",
                max_abs_diff(h_out, h_ref));

    const float ms_f32x4 = benchmark_ms([&]() {
        softmax_f32x4<64><<<rows, 64>>>(d_in, d_out, rows, cols);
    });
    std::printf("softmax_f32x4: %.3f ms\n", ms_f32x4);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
