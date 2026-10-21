%%writefile cute_gemm.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

using namespace cute;

// C (M by N) = A (M by K) * B^T, where B is stored as (N by K). Both have K contiguous.
template <int tileM, int tileN, int tileK, class TiledMma, class TA, class TB, class TC>
__global__ void matmulCute(TiledMma tiled_mma, const TA* A, const TB* B, TC* C,
                           int M, int N, int K) {
  Tensor tA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
  Tensor tB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
  Tensor tC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

  int col = blockIdx.x;
  int row = blockIdx.y;

  Tensor gA = local_tile(tA, make_tile(Int<tileM>{}, Int<tileK>{}), make_coord(row, _));  // (tileM, tileK, k_tile)
  Tensor gB = local_tile(tB, make_tile(Int<tileN>{}, Int<tileK>{}), make_coord(col, _));  // (tileN, tileK, k_tile)
  Tensor gC = local_tile(tC, make_tile(Int<tileM>{}, Int<tileN>{}), make_coord(row, col)); // (tileM, tileN)

  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  Tensor tAgA = thr_mma.partition_A(gA);   // (MMA, MMA_M, MMA_K, k_tile)
  Tensor tBgB = thr_mma.partition_B(gB);   // (MMA, MMA_N, MMA_K, k_tile)
  Tensor tCgC = thr_mma.partition_C(gC);   // (MMA, MMA_M, MMA_N)

  Tensor tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
  Tensor tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  Tensor tCrC = thr_mma.partition_fragment_C(gC);

  clear(tCrC);

  int k_tiles = size<2>(gA);
  #pragma unroll 1
  for (int k = 0; k < k_tiles; ++k) {
    copy(tAgA(_, _, _, k), tArA);
    copy(tBgB(_, _, _, k), tBrB);
    gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  copy(tCrC, tCgC);
}

// Plain reference, one thread per output element, fp32 accumulation.
__global__ void ref_gemm(const half_t* A, const half_t* B, float* C, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m < M && n < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += float(A[m * K + k]) * float(B[n * K + k]);
    C[m * N + n] = acc;
  }
}

int main() {
  constexpr int tileM = 128, tileN = 128, tileK = 32;
  int M = 2048, N = 2048, K = 2048;   // must be multiples of 128, 128, 32

  using MmaAtom = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
  auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _2, _1>>{});  // 2x2 warps = 128 threads

  thrust::host_vector<half_t> h_A(M * K), h_B(N * K);
  for (int i = 0; i < M * K; ++i) h_A[i] = half_t(float((i % 7) - 3) / 4.f);
  for (int i = 0; i < N * K; ++i) h_B[i] = half_t(float((i % 5) - 2) / 4.f);
  thrust::device_vector<half_t> d_A = h_A, d_B = h_B;
  thrust::device_vector<float> d_C(M * N, 0.f), d_R(M * N, 0.f);

  const half_t* pA = thrust::raw_pointer_cast(d_A.data());
  const half_t* pB = thrust::raw_pointer_cast(d_B.data());
  float* pC = thrust::raw_pointer_cast(d_C.data());
  float* pR = thrust::raw_pointer_cast(d_R.data());

  dim3 grid(N / tileN, M / tileM);
  dim3 block(int(size(tiled_mma)));

  auto launch = [&]() {
    matmulCute<tileM, tileN, tileK><<<grid, block>>>(tiled_mma, pA, pB, pC, M, N, K);
  };

  // Correctness
  launch();
  ref_gemm<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(pA, pB, pR, M, N, K);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

  thrust::host_vector<float> h_C = d_C, h_R = d_R;
  int bad = 0; float max_err = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float e = fabsf(h_C[i] - h_R[i]);
    max_err = fmaxf(max_err, e);
    bad += (e > 1e-2f);
  }
  printf("mismatches: %d, max abs error: %g\n", bad, max_err);

  // Timing
  for (int i = 0; i < 3; ++i) launch();   // warmup
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  int iters = 20;
  cudaEventRecord(t0);
  for (int i = 0; i < iters; ++i) launch();
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);
  ms /= iters;
  double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
  printf("avg time: %.3f ms, %.2f TFLOP/s\n", ms, tflops);
}