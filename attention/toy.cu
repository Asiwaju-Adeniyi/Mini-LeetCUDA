#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

// Toy attention with online softmax.
//
// Problem size is intentionally tiny: 2 query rows, 4 keys, head dimension 8.
// One block of 32 threads computes the 2x4 score matrix cooperatively, then
// one thread per query row performs an online-softmax accumulation over keys.
__global__ void toy_attention(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              float* __restrict__ o) {
    constexpr int kQueries = 2;
    constexpr int kKeys = 4;
    constexpr int kHeadDim = 8;

    __shared__ float sQ[kQueries * kHeadDim];
    __shared__ float sK[kKeys * kHeadDim];
    __shared__ float sV[kKeys * kHeadDim];
    __shared__ float sS[kQueries * kKeys];

    for (int i = threadIdx.x; i < kQueries * kHeadDim; i += blockDim.x) {
        sQ[i] = q[i];
    }
    for (int i = threadIdx.x; i < kKeys * kHeadDim; i += blockDim.x) {
        sK[i] = k[i];
        sV[i] = v[i];
    }
    __syncthreads();

    // One thread per (query, key) pair.
    if (threadIdx.x < kQueries * kKeys) {
        const int row = threadIdx.x / kKeys;
        const int key = threadIdx.x % kKeys;

        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            dot += sQ[row * kHeadDim + d] * sK[key * kHeadDim + d];
        }
        sS[row * kKeys + key] = dot * rsqrtf(static_cast<float>(kHeadDim));
    }
    __syncthreads();

    // One thread per query row.  The running state is (max, sum, output).
    if (threadIdx.x < kQueries) {
        const int row = threadIdx.x;
        float running_max = -FLT_MAX;
        float running_sum = 0.0f;
        float acc[kHeadDim] = {0.0f};

        for (int key = 0; key < kKeys; ++key) {
            const float score = sS[row * kKeys + key];
            const float new_max = fmaxf(running_max, score);
            const float scale = __expf(running_max - new_max);
            const float weight = __expf(score - new_max);

            running_sum = running_sum * scale + weight;
            for (int d = 0; d < kHeadDim; ++d) {
                acc[d] = acc[d] * scale + weight * sV[key * kHeadDim + d];
            }
            running_max = new_max;
        }

        for (int d = 0; d < kHeadDim; ++d) {
            o[row * kHeadDim + d] = acc[d] / running_sum;
        }
    }
}

static void cpu_attention(const std::vector<float>& q,
                          const std::vector<float>& k,
                          const std::vector<float>& v, int queries, int keys,
                          int head_dim, std::vector<float>& o) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int row = 0; row < queries; ++row) {
        std::vector<float> scores(keys);
        float max_score = -FLT_MAX;
        for (int key = 0; key < keys; ++key) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q[row * head_dim + d] * k[key * head_dim + d];
            }
            scores[key] = dot * scale;
            max_score = std::max(max_score, scores[key]);
        }

        float sum = 0.0f;
        for (int key = 0; key < keys; ++key) {
            scores[key] = std::exp(scores[key] - max_score);
            sum += scores[key];
        }

        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int key = 0; key < keys; ++key) {
                acc += scores[key] * v[key * head_dim + d];
            }
            o[row * head_dim + d] = acc / sum;
        }
    }
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
    constexpr int queries = 2;
    constexpr int keys = 4;
    constexpr int head_dim = 8;

    std::vector<float> h_q(queries * head_dim);
    std::vector<float> h_k(keys * head_dim);
    std::vector<float> h_v(keys * head_dim);
    std::vector<float> h_o(queries * head_dim);
    std::vector<float> h_ref(queries * head_dim);

    fill_random(h_q);
    fill_random(h_k);
    fill_random(h_v);
    cpu_attention(h_q, h_k, h_v, queries, keys, head_dim, h_ref);

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, h_o.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(32);
    const dim3 grid(1);

    toy_attention<<<grid, block>>>(d_q, d_k, d_v, d_o);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::printf("toy_attention max abs diff: %.6g\n",
                max_abs_diff(h_o, h_ref));

    const float ms = benchmark_ms([&]() {
        toy_attention<<<grid, block>>>(d_q, d_k, d_v, d_o);
    });
    std::printf("toy_attention: %.3f ms\n", ms);

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    return 0;
}
