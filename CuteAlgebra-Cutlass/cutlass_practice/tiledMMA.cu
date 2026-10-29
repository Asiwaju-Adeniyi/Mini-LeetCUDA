#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>

template <int mTile, int nTile, int kTile> 

__global__ void mmaTile(const half *a, const half *b, half *c, int m, int n, int k) {

   Tensor A = make_tensor(make_gmem_ptr(a), make_shape(m, k), make_stride(k, Int<1>{}));
   Tensor B = make_tensor(make_gmem_ptr(b), make_shape(n, k), make_stride(k, Int<1>{}));
   Tensor C = make_tensor(make_gmem_ptr(c), make_shape(m, n), make_stride(n, Int<1>{}));
   
   Tensor lA = local_tile(A, make_tile(Int<mTile>{}, Int<kTile>{}), make_coord(blockIdx.y, _));
   Tensor lB = local_tile(B, make_tile(Int<nTile>{}, Int<kTile>{}), make_coord(blockIdx.x, _));
   Tensor lC = local_tile(C, make_tile(Int<mTile>{}, Int<nTile>{}), make_coord(blockIdx.y, blockIdx.x));

   using mmaOP = SM80_16x8x16_F16F16F16F16_TN;
   using opTrait = Copy_Trait<mmaOP>;
   using opAtom = MMA_Atom<opTrait>;

   static constexpr int Mrepeat = 2;
   static constexpr int Nrepeat = 2;
   static constexpr int Krepeat = 1;
}