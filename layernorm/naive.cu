#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/** input is a matric of dimension N by C;
 * out is also a matrix of dimension N by C;
 *  mean is a separate buffer of size N that stores the mean computed for each row. 
 * The reason we store mean in a buffer despite computing it midkernel is the backward pass — during backprop, LayerNorm's gradient computation needs the mean and rstd from the forward pass. 
 * So I store them now to avoid recomputing later.
 * rstd means reciprocal standard devation which is a representation of 1 / sqrt (var - eps) in the LayerNorm formula.
 * We do this cos division is costly on the GPU, so it's better to do it once, 
 * and then use rstd with the multiplication operator to satisfy the GPU's prefernences.
 * gamma is the learned weight of dimension C (a row vector). One learned scale value per feature dimension. Shared across all N rows.
 * beta (bias) of dimension C. One learned shift per feature dimension. Also shared across all N rows.
 */

__global__ void naiveLayerNorm(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, 
    float* __restrict__ mean, const float* __restrict__ gamma, const float* __restrict__ beta, int N, int C) {

    }