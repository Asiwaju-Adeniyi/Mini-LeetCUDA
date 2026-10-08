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


        Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(N));
        Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(N));
        Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(N));

        Tensor tzr = local_tile(tz, make_shape(Int<elemPerT>{}), make_coord(idx));
        Tensor txr = local_tile(tx, make_shape(Int<elemPerT>{}), make_coord(idx));
        Tensor tyr = local_tile(ty, make_shape(Int<elemPerT>{}), make_coord(idx));
        
        Tensor txR = make_tensor_like(txr);
        Tensor tyR = make_tensor_like(tyr);
        Tensor tzR = make_tensor_like(tzr);

        copy(txr, txR);
        copy(tyr, tyR);
        
        half2 a2 = {a, a};
        half2 b2 = {b, b};
        half2 c2 = {c, c};


        auto tzR2 = recast<half2>(tzR);
        auto txR2 = recast<half2>(txR);
        auto tyR2 = recast<half2>(tyR);

        #pragma unroll 

        for (int i = 0; i < size(tzR2); i++){
            tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
        }

       auto tzRx = recast<half>(tzR2);
        
        copy(tzRz, tzr);
        
    }