// Hopper WGMMA track.
//
// Following Colfax's CUTLASS tutorial series:
//   Part 1 - WGMMA: https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
//   TMA:    https://research.colfax-intl.com/tutorial-hopper-tma/
//   Part 2 - pipelining: https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
//
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
    >{};

auto tiledMMA = make_tiled_mma(mmaAtom{});

template <class TiledMMA, int mTile, int nTile, int kTile, int bP, class T>

__global__ void cutlassGemm(const half_t *a, const half_t* b, half_t* c, int m, int n, int k) {

    //global tensors of each operand

   Tensor A = make_tensor(make_gmem_ptr(a), make_shape(m, k), make_stride(k, Int<1>{}));
   Tensor B = make_tensor(make_gmem_ptr(b), make_shape(n, k), make_stride(k, Int<1>{}));
   Tensor C = make_tensor(make_gmem_ptr(c), make_shape(m, n), make_stride(n, Int<1>{}));

   //CTA-level tiling
   
  Tensor lA = local_tile(A, make_tile(Int<mTile>{}, Int<kTile>{}), make_coord(blockIdx.y, _));
  Tensor lB = local_tile(B, make_tile(Int<nTile>{}, Int<kTile>{}), make_coord(blockIdx.x, _));
  Tensor lC = local_tile(C, make_tile(Int<mTile>{}, Int<nTile>{}), make_coord(blockIdx.y, blockIdx.x));


  //Shared memory layouts 

auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(mTile, kTile, Int<bP>{}));
auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(nTile, kTile, Int<bP>{}));



}
