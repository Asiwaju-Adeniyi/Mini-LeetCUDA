#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

template <tM, tN> 

__global__ void tiledCopyG2S(float *x) {
    __shared__ float sC[tM * tN];

    Tensor gTensor = make_tensor(make_gmem_tensor(x), make_shape(Int<tM>{}, Int<tN>{}), make_stride(Int<tN>{}, Int<1>{}));

    Tensor sTensor = make_tensor(make_smem_tensor(sC), make_shape(Int<tM>{}, Int<tN>{}), make_stride(Int<tN>{}, Int<1>{}));

    using gs2Copy = UniversalCopy<cute::uint64_t>;
    using g2sCopyTraits = Copy_Traits<g2sCopy>;
    using cAtom = Copy_Atom<g2sCopyTraits, float>;
    
    Layout tLayout = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{}));
    Layout vLayout = make_layout(make_shape(Int<1>{}, Int<2>{}));

    auto tCopy = make_tiled_copy(cAtom, tLayout, vLayout);
    
}