#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>

using namespace cute;

  // SM80 16x8x16, fp16 inputs, fp32 accumulate.
using mma_op = SM80_16x8x16_F32F16F16F32_TN;
using mma_traits = MMA_Traits<mma_op>;
using mma_atom = MMA_Atom<mma_traits>;

constexpr int kMmaEURepeatM = 2;
constexpr int kMmaEURepeatN = 2;
constexpr int kMmaEURepeatK = 1;

static constexpr int kMmaPM = 32;
static constexpr int kMmaPN = 32;
static constexpr int kMmaPK = 16;

using MMA_EU_RepeatT = decltype(make_layout(make_shape(
    Int<kMmaEURepeatM>{},
    Int<kMmaEURepeatN>{},
    Int<kMmaEURepeatK>{}
)));

using MMA_P_T = Tile<
    Int<kMmaPM>,
    Int<kMmaPN>,
    Int<kMmaPK>
>;

using MMA = decltype(make_tiled_mma(
    mma_atom{},
    MMA_EU_RepeatT{},
    MMA_P_T{}
));

using TiledMMA = MMA;



// C (M by N) = A (M by K) * B^T, where B is stored as (N by K).
// Every warp builds the same TiledMMA; the block-level tile is 128x128x32 and
// the MMA atom is the SM80 16x8x16 fp16 tensor-core instruction with fp32
// accumulation (so the accumulator matches the fp32 reference below).

template <int mTile, int nTile, int kTile>
__global__ void mmaTile(const half_t* a, const half_t* b, float* c,
                        int m, int n, int k) {
  Tensor A = make_tensor(make_gmem_ptr(a), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(b), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(c), make_shape(m, n), make_stride(n, Int<1>{}));

  Tensor lA = local_tile(A, make_tile(Int<mTile>{}, Int<kTile>{}), make_coord(blockIdx.y, _));
  Tensor lB = local_tile(B, make_tile(Int<nTile>{}, Int<kTile>{}), make_coord(blockIdx.x, _));
  Tensor lC = local_tile(C, make_tile(Int<mTile>{}, Int<nTile>{}), make_coord(blockIdx.y, blockIdx.x));

  TiledMMA tiled_mma;

  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  Tensor tAgA = thr_mma.partition_A(lA);   // (MMA, MMA_M, MMA_K, k_tile)
  Tensor tBgB = thr_mma.partition_B(lB);   // (MMA, MMA_N, MMA_K, k_tile)
  Tensor tCgC = thr_mma.partition_C(lC);   // (MMA, MMA_M, MMA_N)

  Tensor tArA = thr_mma.partition_fragment_A(lA(_, _, 0));
  Tensor tBrB = thr_mma.partition_fragment_B(lB(_, _, 0));
  Tensor tCrC = thr_mma.partition_fragment_C(lC);

  clear(tCrC);

  int k_tiles = size<2>(lA);
  #pragma unroll 1
  for (int kk = 0; kk < k_tiles; ++kk) {
    copy(tAgA(_, _, _, kk), tArA);
    copy(tBgB(_, _, _, kk), tBrB);
    gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  copy(tCrC, tCgC);
}



// Plain reference, one thread per output element, fp32 accumulation.
__global__ void ref_gemm(const half_t* A, const half_t* B, float* C,
                         int M, int N, int K) {
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

  TiledMMA tiledMMA;

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
    mmaTile<tileM, tileN, tileK><<<grid, block>>>(pA, pB, pC, M, N, K);
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
