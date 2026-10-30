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

   static constexpr int MrepeatEU = 2; //MMA repeat frequency for blocks in Operand A and C
   static constexpr int NrepeatEU = 2; //MMA repeat frequency for blocks in Operand B and C
   static constexpr int KrepeatEU = 1; //MMA repeat frequency for blocks in A and B's K dim.


   using mmaAtomShape = opTrait::Shape_MNK;
   static constexpr int kMmaPM = 32;
   static constexpr int kMmaPN = 32;
   static constexpr int kMmaPK = 16;


   using mmaRepeat = decltype(make_layout(make_shape(Int<MrepeatEU>{}, Int<NrepeatEU>{}, Int<KrepeatEU>{})));
   using mmaPT = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

   using MMA = decltype(make_tiled_mma(opAtom{}, mmaRepeat{}, mmaPT));

   MMA tiled_mma;

   auto tMMA = tiled_mma.get_slice(threadIdx.x);
   auto trA = tMMA.partition_fragment_A(lA(_, _, 0));
   auto tgA = tMMA.partition_A(lA(_, _, 0));


   auto trB = tMMA.partition_fragment_A(lB(_, _, 0));
   auto tgB = tMMA.partition_A(lB(_, _, 0));

   auto trC = tMMA.partition_fragment_A(lC);
   auto tgC = tMMA.partition_C(lC);

   cute::copy(tgA, trA);
   cute::copy(tgB, trB);

   cute::gemm(tiled_mma, trA, trB, trC);

   cute::copy(trC, tgC);
   
}