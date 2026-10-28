// cuBLAS reference points for the hand-written GEMMs.
//
// Both products are computed with the same row-major A (M x K) and B (K x N)
// layout as the kernels in this directory.  Since cuBLAS is column-major, we
// compute C^T = B^T * A^T and hand it B, A, C in that order.
//
//   ./cublas_ref              -> 1024^3
//   ./cublas_ref 4096         -> 4096^3
//   ./cublas_ref 4096 2048 512
//
// SGEMM uses the default FP32 math mode (no TF32), so it is a like-for-like
// baseline for the SIMT FP32 kernels.  HGEMM uses FP16 inputs with FP32
// accumulation and FP16 output, which is the tensor-core baseline for the
// SIMT FP16 kernels.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.cuh"

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t status__ = (call);                                   \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                            \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,   \
                         __LINE__, static_cast<int>(status__));             \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

int main(int argc, char** argv) {
    int M, N, K;
    parse_gemm_dims(argc, argv, M, N, K);

    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c(M * N);
    fill_random(h_a);
    fill_random(h_b);

    std::vector<__half> h_a_half(M * K);
    std::vector<__half> h_b_half(K * N);
    std::vector<__half> h_c_half(M * N);
    for (int i = 0; i < M * K; ++i) {
        h_a_half[i] = __float2half(h_a[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_b_half[i] = __float2half(h_b[i]);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    __half* d_a_half = nullptr;
    __half* d_b_half = nullptr;
    __half* d_c_half = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a_half, h_a_half.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b_half, h_b_half.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_c_half, h_c_half.size() * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_half, h_a_half.data(),
                          h_a_half.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_half, h_b_half.data(),
                          h_b_half.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --- SGEMM -------------------------------------------------------------
    auto launch_sgemm = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha, d_b, N, d_a, K, &beta, d_c, N));
    };

    launch_sgemm();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int bad = verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-3f);
    std::printf("cublas_sgemm mismatches: %d\n", bad);

    float ms = benchmark_ms(launch_sgemm);
    std::printf("cublas_sgemm: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    // --- HGEMM ---
    auto launch_hgemm = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b_half,
            CUDA_R_16F, N, d_a_half, CUDA_R_16F, K, &beta, d_c_half,
            CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    launch_hgemm();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c_half.data(), d_c_half,
                          h_c_half.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        h_c[i] = __half2float(h_c_half[i]);
    }
    bad = verify_gemm_sample(h_a, h_b, h_c, M, N, K, 32, 1e-2f);
    std::printf("cublas_hgemm mismatches: %d\n", bad);

    ms = benchmark_ms(launch_hgemm);
    std::printf("cublas_hgemm: %.3f ms, %.2f GFLOP/s\n", ms,
                gflops(M, N, K, ms));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_a_half));
    CUDA_CHECK(cudaFree(d_b_half));
    CUDA_CHECK(cudaFree(d_c_half));
    return 0;
}
