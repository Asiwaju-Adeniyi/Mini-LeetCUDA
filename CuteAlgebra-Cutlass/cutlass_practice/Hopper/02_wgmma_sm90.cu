// WGMMA GEMM: TMA. 

#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>
 #include <cutlass/arch/barrier.h>

using namespace cute;

//Writing out the TMA code in the Host before using it in the kernel. 

template<typename T, typename I, typename BM, class BN> 

void host_CTA(T* data, I M, I N) {

    //global memory tensor object creation
    auto gLayout = make_layout(make_shape(M, N), LayoutRight{});
    auto gTensor = make_tensor(make_gmem_ptr(T), gLayout);

     //shared memory tensor object creation
    auto sLayout = make_layout(make_shape(BM, BN), LayoutRight{});

    //create the TMA load object that takes global and shared memory tensor objects as parameters 
    auto tmaLoad = make_tma_copy(SM90_TMA_LOAD{}, gTensor, sLayout);

    //call the function

    tmaLoadFunction<BM, BN> <<<dim3{M/BM, N/BN, 1}>>>(tmaLoad, gTensor, sLayout);
}


