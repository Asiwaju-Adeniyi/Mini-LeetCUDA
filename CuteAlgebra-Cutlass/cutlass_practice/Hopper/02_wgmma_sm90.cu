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

