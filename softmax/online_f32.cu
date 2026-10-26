#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "softmax_common.cuh"

// Online row-wise softmax: each thread maintains (max, denominator), then the
// block combines the partial states with the online-softmax merge rule.
template <const int NumThreads = 256>
__global__ void online_softmax_f32(const float* __restrict__ input,
                                   float* __restrict__ output, int rows,
                                   int cols) {
    const int row = blockIdx.x;
    const float* x = input + row * cols;
    float* y = output + row * cols;

    MD val{-FLT_MAX, 0.0f};
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float xi = x[i];
        if (xi > val.M) {
            val.D = val.D * __expf(val.M - xi) + 1.0f;
            val.M = xi;
        } else {
            val.D += __expf(xi - val.M);
        }
    }

    const MD total = block_reduce_md<NumThreads>(val);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = __expf(x[i] - total.M) / total.D;
    }
}

static std::vector<float> cpu_safe_softmax(const std::vector<float>& input,
                                           int rows, int cols) {
    std::vector<float> output(input.size());
    for (int r = 0; r < rows; ++r) {
        float max_val = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            max_val = std::max(max_val, input[r * cols + c]);
        }
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum += std::exp(static_cast<double>(input[r * cols + c] - max_val));
        }
        for (int c = 0; c < cols; ++c) {
            output[r * cols + c] = static_cast<float>(
                std::exp(static_cast<double>(input[r * cols + c] - max_val)) /
                sum);
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

    std::vector<float> h_in(rows * cols);
    std::vector<float> h_out(rows * cols);
    fill_random(h_in, -1.0f, 1.0f);
    const std::vector<float> h_ref = cpu_safe_softmax(h_in, rows, cols);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, h_out.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    online_softmax_f32<256><<<rows, 256>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::printf("online_softmax_f32 max abs diff: %.6g\n",
                max_abs_diff(h_out, h_ref));

    const float ms = benchmark_ms([&]() {
        online_softmax_f32<256><<<rows, 256>>>(d_in, d_out, rows, cols);
    });
    std::printf("online_softmax_f32: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
