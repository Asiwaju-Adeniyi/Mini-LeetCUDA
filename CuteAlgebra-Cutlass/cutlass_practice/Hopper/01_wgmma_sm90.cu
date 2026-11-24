// Scaffold only. The first kernel to implement here is a single-stage SM90
// WGMMA GEMM; TMA and pipelining follow in later files.

#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>
 #include <cutlass/arch/barrier.h>

using namespace cute;


using mmaAtom =
    SM90_64x64x16_F16F16F16_SS<
        GMMA::Major::MN,
        GMMA::Major::MN
    >;

using TiledMma = decltype(make_tiled_mma(mmaAtom{}));

//Array Engine to accomodate pipeline stages (using just 1 pipeline stage for learning.)

template <class T, class LayoutA, class LayoutB> 

struct SharedStorage{
    alignas(128) ArrayEngine<T, cosize_v<LayoutA>> A;
    alignas(128) ArrayEngine<T, cosize_v<LayoutB>> B;
};


template <int mTile, int nTile, int kTile, int bP, class T>

__global__ void cutlassGemm(TiledMma mma, const half_t *a, const half_t* b, half_t* c, int m, int n, int k) {

    //global tensors of each operand

   Tensor A = make_tensor(make_gmem_ptr(a), make_shape(m, k), make_stride(k, Int<1>{}));
   Tensor B = make_tensor(make_gmem_ptr(b), make_shape(n, k), make_stride(k, Int<1>{}));
   Tensor C = make_tensor(make_gmem_ptr(c), make_shape(m, n), make_stride(n, Int<1>{}));

   //CTA-level tiling
   
  Tensor lA = local_tile(A, make_tile(Int<mTile>{}, Int<kTile>{}), make_coord(blockIdx.y, _));
  Tensor lB = local_tile(B, make_tile(Int<nTile>{}, Int<kTile>{}), make_coord(blockIdx.x, _));
  Tensor lC = local_tile(C, make_tile(Int<mTile>{}, Int<nTile>{}), make_coord(blockIdx.y, blockIdx.x));


  //Shared memory layouts 

auto sA_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(mTile, kTile, Int<bP>{}));
auto sB_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(nTile, kTile, Int<bP>{}));


//Shared memory declarations
using SmemLayoutA = decltype(sA_layout);
using SmemLayoutB = decltype(sB_layout);

extern __shared__ char sharedMem[];

using Storage = SharedStorage<T, SmemLayoutA, SmemLayoutB>;

Storage& storage = *reinterpret_cast<Storage*>(sharedMem);

Tensor sA = make_tensor(make_smem_ptr(storage.A.begin()), sA_layout);
Tensor sB = make_tensor(make_smem_ptr(storage.B.begin()), sB_layout);

// partitioning work for shared memory copy

// Making A and B copy atoms
auto copyA = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{}, 
    Layout<Shape<_4, _32>>{}, Layout<Shape<_1, _1>>{});
auto copyB = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{}, 
    Layout<Shape<_4, _32>>{}, Layout<Shape<_1, _1>>{});

//Making A and B thread layouts, defining the values threads will copy, and partitioning the source and destination tensors. 
auto thr_copy_A = copyA.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_A.partition_S(lA);
Tensor tAsA = thr_copy_A.partition_D(sA);
Tensor tArA = make_fragment_like(tAsA(_,_,_,0));

auto thr_copy_B = copyB.get_slice(threadIdx.x);
Tensor tBgB = thr_copy_B.partition_S(lB);
Tensor tBsB = thr_copy_B.partition_D(sB);
Tensor tBrB = make_fragment_like(tBsB(_,_,_,0));


// partitioning work for mma
auto tMMA = mma.get_slice(threadIdx.x);

Tensor partA = tMMA.partition_A(sA);
Tensor partB = tMMA.partition_B(sB);
Tensor partC = tMMA.partition_C(lC);

Tensor fragA = tMMA.make_fragment_A(partA);
Tensor fragB = tMMA.make_fragment_B(partB);
Tensor fragC = tMMA.make_fragment_C(partC);

clear(fragC);


auto step = size<2>(lA);

for (int i = 0; i < step; i++) 
{
    //We first copy from global to registers then copy from registers to shared.
    copy(copyA, tAgA(_,_,_,i), tArA);
    copy(copyB, tAgB(_,_,_,i), tBrB);
    copy(tArA, tAsA(_,_,_,0));  //We only have one shared memory staging buffer hence the 0.
    copy(tBrB, tBsB(_,_,_,0)); 

    __syncthreads(); //thread synchronization per block.
    
    cutlass::arch::fence_view_async_shared(); //warp group synchronization primitive. 


    cute::warpgroup_fence_operand(fragC);

    cute::warpgroup_arrive();

    gemm(mma, fragA(_,_,_,0), fragB(_,_,_,0), fragC);


    cute::warpgroup_commit_batch();

    cute::warpgroup_wait<0>();

    cute::warpgroup_fence_operand(fragC);
    
};

    copy(fragC, partC);

}


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
  constexpr int tileM = 128, tileN = 128, tileK = 32, stages = 1;
  int M = 2048, N = 2048, K = 2048;

  TiledMma tiled_mma;

  thrust::host_vector<half_t> h_A(M * K), h_B(N * K);
  for (int i = 0; i < M * K; ++i) h_A[i] = half_t(float((i % 7) - 3) / 4.f);
  for (int i = 0; i < N * K; ++i) h_B[i] = half_t(float((i % 5) - 2) / 4.f);
  thrust::device_vector<half_t> d_A = h_A, d_B = h_B;
  thrust::device_vector<half_t> d_C(M * N, half_t(0));
  thrust::device_vector<float> d_R(M * N, 0.f);

  const half_t* pA = thrust::raw_pointer_cast(d_A.data());
  const half_t* pB = thrust::raw_pointer_cast(d_B.data());
  half_t* pC = thrust::raw_pointer_cast(d_C.data());
  float* pR = thrust::raw_pointer_cast(d_R.data());

  auto sA_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<half_t>{},
                                 make_shape(Int<tileM>{}, Int<tileK>{}, Int<stages>{}));
  auto sB_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<half_t>{},
                                 make_shape(Int<tileN>{}, Int<tileK>{}, Int<stages>{}));
  int smem_bytes = int(cosize(sA_layout) + cosize(sB_layout)) * int(sizeof(half_t));

  dim3 grid(N / tileN, M / tileM);
  dim3 block(int(size(tiled_mma)));

  auto launch = [&]() {
    cutlassGemm<tileM, tileN, tileK, stages, half_t>
      <<<grid, block, smem_bytes>>>(tiled_mma, pA, pB, pC, M, N, K);
  };

  launch();
  ref_gemm<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(pA, pB, pC, pR, M, N, K);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

  thrust::host_vector<half_t> h_C = d_C;
  thrust::host_vector<float> h_R = d_R;
  int bad = 0; float max_err = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float e = fabsf(float(h_C[i]) - h_R[i]);
    max_err = fmaxf(max_err, e);
    bad += (e > 1e-2f);
  }
  printf("mismatches: %d, max abs error: %g\n", bad, max_err);

  for (int i = 0; i < 3; ++i) launch();
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
