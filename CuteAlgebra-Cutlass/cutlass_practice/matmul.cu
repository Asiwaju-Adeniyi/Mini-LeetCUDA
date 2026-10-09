#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


template <const int elemPerT = 8, const int M, const int N, const int K> 

__global__ void testCuteMatmul (float *c, const float *a, const float *b, int N) {
  auto mA = make_tensor(make_gmem_ptr(a), make_shape(M, K), Int<1>{}, M);
  auto mB = make_tensor(make_gmem_ptr(b), make_shape(N, K), Int<1>{}, N);
  auto mC = make_tensor(make_gmem_ptr(c), make_shape(M, N), Int<1>{}, K);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  

  // Shared memory buffers
__shared__ TA smemA[bM * bK];
__shared__ TB smemB[bN * bK];
auto sA = make_tensor(make_smem_ptr(smemA), make_layout(make_shape(bM,bK)));
auto sB = make_tensor(make_smem_ptr(smemB), make_layout(make_shape(bN,bK)));  


}