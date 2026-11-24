// Scaffold only. The first kernel to implement here is a single-stage SM90
// WGMMA GEMM; TMA and pipelining follow in later files.

#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>

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

auto tMMA = mma.get_slice(threadIdx.x);

Tensor partA = tMMA.partition_A(sA);
Tensor partB = tMMA.partition_B(sB);
Tensor partC = tMMA.partition_C(lC);

Tensor fragA = tMMA.make_fragment_A(partA(_,_, 0));
Tensor fragB = tMMA.make_fragment_B(partB(_,_,0));
Tensor fragC = tMMA.make_fragment_C(partC);

clear(fragC);

auto step = size<2>(lA);

for (int i = 0; i < step; i++) 
{

    copy(lA(_,_, i), sA(_, _, 0));
    copy(lB(_,_, i), sB(_, _, 0));

    __syncthreads();

__syncthreads();

fence_view_async_shared();

    cute::warpgroup_fence_operand(fragC);

    cute::warpgroup_arrive();

    gemm(mma, fragA(_,_,_,0), fragB(_,_,_,0), fragC);


    cute::warpgroup_commit_batch();

    cute::warpgroup_wait<0>();

    cute::warpgroup_fence_operand(fragC);
    
};

    copy(fragC, lC);

}
