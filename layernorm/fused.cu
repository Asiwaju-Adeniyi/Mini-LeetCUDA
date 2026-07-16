#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WarpSize 32

struct __align__(8) state{
    float sum;
    float sumSQ;
};

template <const int kW = WarpSize>
__device__ __forceinline__ float warpReduc(float val) {
    #pragma unroll 
    for (int offset = kW >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <const int numThreads = 256> 

__device__ __forceinline__ float blockReduc(float val, float* reducBuf) {
    
      constexpr int warpNum = (numThreads + WarpSize - 1) / WarpSize;
      int warp = threadIdx.x / WarpSize;
      int lane = threadIdx.x % WarpSize;


      val = warpReduc<WarpSize>(val);

      if (lane == 0) {
        reducBuf[warp] = val;
      };
      __syncthreads();

      val = (lane < warpNum) ? reducBuf[lane] : 0.0f;

      if (warp == 0){
        val = warpReduc<WarpNume>(val);
      } 

      return val;
    
    }


// __global__ void naiveFusedLayerNorm(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, 
//     float* __restrict__ mean, const float* __restrict__ gamma, const float* __restrict__ beta, int N, int C) {

//         int tid = threadIdx.x + blockDim.x * blockIdx.x;
//         float eps = 1e-5f;

//         if (tid >= N) {return; };

//         state fused{0.0f, 0.0f};
//         const float* x = inp + tid * C;

//         for (int i = 0; i < C; i++) {
//         fused.sum += x[i];
//         fused.sumSQ += x[i] * x[i];
//         }

//         float m  = fused.sum / C;
//         float var = (fused.sumSQ / C) - (m * m);

//         float s = rsqrtf(var + eps);

//         float* y = out + tid * C;

//         for (int i = 0; i < C; i++) {
//             float norm = (x[i] - m) * s;
//             float scaled = gamma[i] * norm + beta[i];

//             y[i] = scaled;
//         }

//         mean[tid] = m;
//         rstd[tid] = s;

//     }

template <const int numThreads = 256> 

__global__ void blockFusedLayerNorm(const float* __restrict__ inp, float* __restrict__ out, float* __restrict__ rstd, 
    float* __restrict__ mean, const float* __restrict__ gamma, const float* __restrict__ beta, int N, int C) {
        int idx = blockIdx.x;
        float eps = 1e-5f;
         
        const float* x = inp + idx * C;

        state fused{0.0f, 0.0f};

        extern __shared__ float shared[];
         
        float* rowData = shared;           
        float* reducBuf = shared + C;     

        if (idx >= N) {return;};

        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            shared[i] = x[i];
        }
        __syncthreads();

        for (int j = threadIdx.x; j < C; j+=blockDim.x) {
            fused.sum += shared[j];
            fused.sumSQ += shared[j] * shared[j];
        }
        
        fused.sum = blockReduc<numThreads>(fused.sum, reducBuf);
        fused.sumSQ = blockReduc<numThreads>(fused.sumSQ, reducBuf);

        float m = fused.sum / C;
        float var = (fused.sumSQ / C) - (m * m); 
        float r = rsqrtf(var + eps);

        float* y = out + idx * C;

        for (int k = threadIdx.x; k < C; k+=blockDim.x) {
            float norm = (shared[k] - m) * r;
            float scaled = gamma[k] * norm + beta[k];
            y[k] = scaled;
        }
       mean[idx] = m;
       rstd[idx] = r;

    }