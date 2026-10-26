#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

#include "../common/cuda_utils.cuh"

struct IndexValue {
    float value;
    int index;

    __device__ __forceinline__ bool operator>(const IndexValue& other) const {
        return value > other.value;
    }
    __device__ __forceinline__ bool operator<(const IndexValue& other) const {
        return value < other.value;
    }
};

__device__ __forceinline__ void heapify_down(IndexValue* heap, int root,
                                             int size) {
    int current = root;
    while (true) {
        int smallest = current;
        const int left = 2 * current + 1;
        const int right = 2 * current + 2;

        if (left < size && heap[left] < heap[smallest]) {
            smallest = left;
        }
        if (right < size && heap[right] < heap[smallest]) {
            smallest = right;
        }
        if (smallest == current) {
            break;
        }

        const IndexValue tmp = heap[current];
        heap[current] = heap[smallest];
        heap[smallest] = tmp;
        current = smallest;
    }
}

// Min-heap of size K per row.  The root is the smallest of the current top-K.
__global__ void topK_Heap(const float* __restrict__ input,
                          float* __restrict__ values,
                          int* __restrict__ indices, int rows, int K, int N) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* x = input + row * N;
    float* v = values + row * K;
    int* idx = indices + row * K;

    extern __shared__ IndexValue heap[];

    if (threadIdx.x < K) {
        heap[threadIdx.x].value = x[threadIdx.x];
        heap[threadIdx.x].index = threadIdx.x;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = K / 2 - 1; i >= 0; --i) {
            heapify_down(heap, i, K);
        }

        for (int i = K; i < N; ++i) {
            if (x[i] > heap[0].value) {
                heap[0].value = x[i];
                heap[0].index = i;
                heapify_down(heap, 0, K);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < K) {
        v[threadIdx.x] = heap[threadIdx.x].value;
        idx[threadIdx.x] = heap[threadIdx.x].index;
    }
}

int main() {
    constexpr int rows = 128;
    constexpr int N = 1024;
    constexpr int K = 8;

    std::vector<float> h_in(rows * N);
    std::vector<float> h_values(rows * K);
    std::vector<int> h_indices(rows * K);
    fill_random(h_in);

    float* d_in = nullptr;
    float* d_values = nullptr;
    int* d_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values, h_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    topK_Heap<<<rows, K, K * sizeof(IndexValue)>>>(d_in, d_values, d_indices,
                                                    rows, K, N);
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
        std::vector<std::pair<float, int>> got;
        got.reserve(K);
        for (int j = 0; j < K; ++j) {
            got.emplace_back(h_values[r * K + j], h_indices[r * K + j]);
        }
        std::sort(got.begin(), got.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.first > rhs.first;
                  });

        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + K, order.end(),
                          [&](int lhs, int rhs) {
                              return h_in[r * N + lhs] > h_in[r * N + rhs];
                          });

        for (int j = 0; j < K; ++j) {
            const float ref = h_in[r * N + order[j]];
            if (order[j] != got[j].second ||
                std::fabs(ref - got[j].first) > 1e-6f) {
                ++bad;
                break;
            }
        }
    }
    std::printf("topK_Heap mismatched rows: %d/%d\n", bad, rows);

    const float ms = benchmark_ms([&]() {
        topK_Heap<<<rows, K, K * sizeof(IndexValue)>>>(
            d_in, d_values, d_indices, rows, K, N);
    });
    std::printf("topK_Heap: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));
    return 0;
}
