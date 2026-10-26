#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

// Naive LayerNorm: one thread per row.
__global__ void naiveLayerNorm(const float* __restrict__ input,
                               float* __restrict__ output,
                               float* __restrict__ rstd,
                               float* __restrict__ mean,
                               const float* __restrict__ gamma,
                               const float* __restrict__ beta, int rows,
                               int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) {
        return;
    }

    const float eps = 1e-5f;
    const float* x = input + idx * cols;

    float m = 0.0f;
    for (int i = 0; i < cols; ++i) {
        m += x[i];
    }
    m /= static_cast<float>(cols);

    float var = 0.0f;
    for (int i = 0; i < cols; ++i) {
        const float temp = x[i] - m;
        var += temp * temp;
    }
    var /= static_cast<float>(cols);

    const float s = rsqrtf(var + eps);

    float* y = output + idx * cols;
    for (int i = 0; i < cols; ++i) {
        y[i] = gamma[i] * ((x[i] - m) * s) + beta[i];
    }

    mean[idx] = m;
    rstd[idx] = s;
}

struct HostRef {
    std::vector<float> output;
    std::vector<float> mean;
    std::vector<float> rstd;
};

static HostRef cpu_layer_norm(const std::vector<float>& input,
                              const std::vector<float>& gamma,
                              const std::vector<float>& beta, int rows,
                              int cols) {
    HostRef ref;
    ref.output.resize(input.size());
    ref.mean.resize(rows);
    ref.rstd.resize(rows);

    const float eps = 1e-5f;
    for (int r = 0; r < rows; ++r) {
        const float* x = input.data() + r * cols;
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum += x[c];
        }
        const double m = sum / cols;

        double var = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double t = x[c] - m;
            var += t * t;
        }
        var /= cols;
        const double s = 1.0 / std::sqrt(var + eps);

        for (int c = 0; c < cols; ++c) {
            ref.output[r * cols + c] = static_cast<float>(
                gamma[c] * ((x[c] - m) * s) + beta[c]);
        }
        ref.mean[r] = static_cast<float>(m);
        ref.rstd[r] = static_cast<float>(s);
    }
    return ref;
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
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);
    fill_random(h_in, -1.0f, 1.0f);
    fill_random(h_gamma, 0.5f, 1.5f);
    fill_random(h_beta, -0.5f, 0.5f);

    const HostRef ref = cpu_layer_norm(h_in, h_gamma, h_beta, rows, cols);

    std::vector<float> h_out(rows * cols);
    std::vector<float> h_mean(rows);
    std::vector<float> h_rstd(rows);

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_mean = nullptr;
    float* d_rstd = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, h_out.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, h_mean.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, h_rstd.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, h_gamma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, h_beta.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr int NumThreads = 256;
    const dim3 block(NumThreads);
    const dim3 grid(ceil_div(rows, NumThreads));

    naiveLayerNorm<<<grid, block>>>(d_in, d_out, d_rstd, d_mean, d_gamma,
                                    d_beta, rows, cols);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, h_mean.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, h_rstd.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::printf("naiveLayerNorm output max abs diff: %.6g\n",
                max_abs_diff(h_out, ref.output));
    std::printf("naiveLayerNorm mean max abs diff:   %.6g\n",
                max_abs_diff(h_mean, ref.mean));
    std::printf("naiveLayerNorm rstd max abs diff:   %.6g\n",
                max_abs_diff(h_rstd, ref.rstd));

    const float ms = benchmark_ms([&]() {
        naiveLayerNorm<<<grid, block>>>(d_in, d_out, d_rstd, d_mean, d_gamma,
                                        d_beta, rows, cols);
    });
    std::printf("naiveLayerNorm: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_rstd));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    return 0;
}
