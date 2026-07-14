#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define warpSize 32
constexpr int WarpNum = (NumThreads + WarpSize - 1) / WarpSize;

template <typename numThreads 256> 

struct __align__(8) state{
    float sum;
    float sumSQ;
};

__device__ __forceinline__ float warpReduc(float val) {
    #pragma unroll 
    for (int offset = WarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduc(float val, int N) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      int warp = threadIdx.x / warpSize;
      int lane = threadIdx.x % warpSize;
      


      __shared__ float reduc[WarpNum];

      float a = (idx < N) ? val[idx] : 0.0f;

      a = warpReduc<warpNum>(a);

      if (lane == 0) {
        reduc[warp] = a;
      };

      __syncthreads();

      a = (lane < warpNum) ? reduc[lane] : 0.0f;

      if (warp == 0){
        a = warpReduc<warpSize>a;
      } 


    
    }

__global__ void fusedLayerNorm(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, 
    float* __restrict__ mean, const float* __restrict__ gamma, const float* __restrict__ beta, int N, int C) {

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        float eps = 1e-5f;

        if (tid >= N) {return; };

        state fused{0.0f, 0.0f};
        const float* x = inp + tid * C;

        for (int i = 0; i < C; i++) {
        fused.sum += x[i];
        fused.sumSQ += x[i] * x[i];
        }

        float m  = fused.sum / C;
        float var = (fused.sumSQ / C) - (m * m);

        float s = rsqrtf(var + eps);

        float* y = out + tid * C;

        for (int i = 0; i < C; i++) {
            float norm = (x[i] - m) * s;
            float scaled = gamma[i] * norm + beta[i];

            y[i] = scaled;
        }

        mean[tid] = m;
        rstd[tid] = s;

    }



__global__ void fusedLayerNorm(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, 
    float* __restrict__ mean, const float* __restrict__ gamma, const float* __restrict__ beta, int N, int C) 