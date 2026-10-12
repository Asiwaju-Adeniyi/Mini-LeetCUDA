
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <class T, int tileM, int tileN, int tileK> 

__global__ void matmulCute(const T* A, T* B, T *C, int M, int N, int K) {
    
    Tensor tA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor tB = make_tensor(make_gmem_ptr(A), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor tC = make_tensor(make_gmem_ptr(A), make_shape(M, N), make_stride(N, Int<1>{}));

    
    int col = blockIdx.x;
    int row = blockIdx.y;

    Tensor tAtile = local_tile(A, make_tile(Int<tileM>{}, Int<tileK>{}), make_coord(row, _));
    Tensor tBtile = local_tile(B, make_tile(Int<tileN>{}, Int<tileK>{}), make_coord(col, _));
    Tensor tCtile = local_tile(C, make_tile(Int<tileM>{}, Int<tileN>{}), make_coord(row,col));

    TiledMMA tiledMMA;

    auto tr_mma = tiledMMA.get_slice(threadIdx.x);

    auto gtA = tr_mma.partition_A(tAtile);
    auto gtB = tr_mma.partition_B(tBtile);
    auto gtC = tr_mma.partition_C(tCtile);


    auto gtAF = tr_mma.partition_fragment_A(gtA(_, _, 0));
    auto gtBF = tr_mma.partition_fragment_B(gtB(_, _, 0));
    auto gtCF = tr_mma.partition_fragment_C(gtC(_, _, 0));

    clear(gtCF);

    int numTileK = size<2>(gA);

    #pragma unroll 1

     for (int i = 0; i < numTileK; ++i) {
      cute::copy(gtA(_, _, _, i), gtAF);
      cute::copy(gtB(_, _, _, i), gtBF);
      
      cute::gemm(tiled_mma, gtCF, gtAF, gtBF, gtCF);
     }

     cute::copy(gtCF, gtC);
}