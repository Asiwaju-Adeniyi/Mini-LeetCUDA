#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "../common/cuda_utils.cuh"

// Single-thread top-k per row, kept as the simple reference implementation.
__global__ void topk(const float* __restrict__ input,
                     float* __restrict__ values, int* __restrict__ indices,
                     int batch_size, int n, int k) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    const float* input_row = input + batch_idx * n;
    float* v_row = values + batch_idx * k;
    int* i_row = indices + batch_idx * k;

    for (int i = 0; i < k; ++i) {
        v_row[i] = -FLT_MAX;
        i_row[i] = -1;
    }

    for (int i = 0; i < n; ++i) {
        const float val = input_row[i];
        if (val <= v_row[k - 1]) {
            continue;
        }

        int j = k - 1;
        while (j > 0 && val > v_row[j - 1]) {
            v_row[j] = v_row[j - 1];
            i_row[j] = i_row[j - 1];
            --j;
        }
        v_row[j] = val;
        i_row[j] = i;
    }
}

int main() {
    constexpr int batch_size = 128;
    constexpr int n = 1024;
    constexpr int k = 8;

    std::vector<float> h_in(batch_size * n);
    std::vector<float> h_values(batch_size * k);
    std::vector<int> h_indices(batch_size * k);
    fill_random(h_in);

    float* d_in = nullptr;
    float* d_values = nullptr;
    int* d_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values, h_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    topk<<<batch_size, 1>>>(d_in, d_values, d_indices, batch_size, n, k);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values,
                          h_values.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices,
                          h_indices.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int b = 0; b < batch_size; ++b) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int lhs, int rhs) {
                              return h_in[b * n + lhs] > h_in[b * n + rhs];
                          });
        for (int j = 0; j < k; ++j) {
            const float ref = h_in[b * n + order[j]];
            const float got = h_values[b * k + j];
            if (order[j] != h_indices[b * k + j] ||
                std::fabs(ref - got) > 1e-6f) {
                ++bad;
                break;
            }
        }
    }
    std::printf("topk mismatched rows: %d/%d\n", bad, batch_size);

    const float ms = benchmark_ms([&]() {
        topk<<<batch_size, 1>>>(d_in, d_values, d_indices, batch_size, n, k);
    });
    std::printf("topk: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));
    return 0;
}
