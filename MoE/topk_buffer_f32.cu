#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "../common/cuda_utils.cuh"

// Selection-buffer top-k: mark selected elements, then pick the next maximum.
__global__ void topkBuffer(const float* __restrict__ input,
                           float* __restrict__ values,
                           int* __restrict__ indices, int rows, int n, int k,
                           bool* __restrict__ selected) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* x = input + row * n;
    float* v = values + row * k;
    int* idx = indices + row * k;
    bool* sel = selected + row * n;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sel[i] = false;
    }
    __syncthreads();

    for (int j = 0; j < k; ++j) {
        if (threadIdx.x == 0) {
            float max_val = -FLT_MAX;
            int max_idx = -1;
            for (int i = 0; i < n; ++i) {
                if (!sel[i] && x[i] > max_val) {
                    max_val = x[i];
                    max_idx = i;
                }
            }
            if (max_idx >= 0) {
                sel[max_idx] = true;
                v[j] = max_val;
                idx[j] = max_idx;
            }
        }
        __syncthreads();
    }
}

int main() {
    constexpr int rows = 128;
    constexpr int n = 1024;
    constexpr int k = 8;
    constexpr int NumThreads = 256;

    std::vector<float> h_in(rows * n);
    std::vector<float> h_values(rows * k);
    std::vector<int> h_indices(rows * k);
    fill_random(h_in);

    float* d_in = nullptr;
    float* d_values = nullptr;
    int* d_indices = nullptr;
    bool* d_selected = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values, h_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_selected, static_cast<size_t>(rows) * n * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    topkBuffer<<<rows, NumThreads>>>(d_in, d_values, d_indices, rows, n, k,
                                     d_selected);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values,
                          h_values.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices,
                          h_indices.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int r = 0; r < rows; ++r) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int lhs, int rhs) {
                              return h_in[r * n + lhs] > h_in[r * n + rhs];
                          });
        for (int j = 0; j < k; ++j) {
            const float ref = h_in[r * n + order[j]];
            const float got = h_values[r * k + j];
            if (order[j] != h_indices[r * k + j] ||
                std::fabs(ref - got) > 1e-6f) {
                ++bad;
                break;
            }
        }
    }
    std::printf("topkBuffer mismatched rows: %d/%d\n", bad, rows);

    const float ms = benchmark_ms([&]() {
        topkBuffer<<<rows, NumThreads>>>(d_in, d_values, d_indices, rows, n, k,
                                         d_selected);
    });
    std::printf("topkBuffer: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_selected));
    return 0;
}
