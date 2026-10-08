#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template < int elemPerT = 8> 

// z = ax + by + c

__global__ void vecAddTiledMultiThread(int N, half *z, const half *x, const half *y, 
    const half b, const half a, const half c) {

        using namespace cute;

        int idx = threadIdx.x + blockIdx.x * blockDim.x; 

        if (idx >= N / elemPerT) {return;}


        Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
        Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
        Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

        Tensor tzr = local_tile(tz, make_shape(Int<elemPerT>{}), make_coord(idx));
        Tensor txr = local_tile(tx, make_shape(Int<elemPerT>{}), make_coord(idx));
        Tensor tyr = local_tile(ty, make_shape(Int<elemPerT>{}), make_coord(idx));
        

        
    }