#include <float.h>
#include <cuda_runtime.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>


__global__ void dummy(const float *Q, const float *K, const float *V, float *O){
 int tid = threadIdx.x;

   __shared__ float sQ[16];
   __shared__ float sK[16];
   __shared__ float sS[4];
   __shared__ float sV[16];
   int rowStart{0}; 

   float rowMax = -INFINITY;
   float rowSum = {}; 
   float A[8] = {};
    
        sQ[rowStart * 8 + tid] = Q[(rowStart) * 8 + tid];
        sQ[(rowStart + 1) * 8 + tid] = Q[(rowStart + 1) * 8 + tid];
        __syncthreads();
 

  for(int kRowStart = 0; kRowStart < 4; kRowStart += 2) {
    float accum = {};
    int tRow = threadIdx.x / 2;
    int tCol = threadIdx.x % 2;

        sK[(rowStart) * 8 + tid] = K[kRowStart * 8 + tid];
        sK[(rowStart + 1) * 8 + tid] = K[(kRowStart + 1) * 8 + tid];

        sV[rowStart * 8 + tid] = V[kRowStart * 8 + tid];
        sV[(rowStart + 1) * 8 + tid] = V[(kRowStart + 1) * 8 + tid];

         __syncthreads();

         if (tid < 4) {for (int i = 0; i < 8; i++) {
               accum += sQ[tRow * 8 + i] * sK[tCol * 8 + i];
         }
         
         sS[tRow * 2 + tCol] = accum;
         };

          __syncthreads();

          if (tid == 0) {

                float oldMax = rowMax;
                float tileMax = fmaxf(sS[0], sS[1]);
                float newMax = fmaxf(oldMax, tileMax);
                float scale = expf(oldMax - newMax);

                rowSum = rowSum * scale + expf(sS[0] - newMax) + expf(sS[1] - newMax);

                for (int d = 0; d < 8; d++) {
                    A[d] = A[d] * scale + (expf(sS[0] - newMax) * sV[d]) + (expf(sS[1] - newMax) * sV[d + 8]);
                }

                rowMax = newMax;


          }

          if (tid == 2){
                float oldMax = rowMax;
                float tileMax = fmaxf(sS[2], sS[3]);
                float newMax = fmaxf(oldMax, tileMax);
                float scale = expf(oldMax - newMax);

                rowSum = rowSum * scale + expf(sS[2] - newMax) + expf(sS[3] - newMax);

                    for (int d = 0; d < 8; d++) {
                    A[d] = A[d] * scale + (expf(sS[2] - newMax) * sV[d]) + (expf(sS[3] - newMax) * sV[d + 8]);
                    
            }
            rowMax = newMax;

          }

          }

            __syncthreads();

if (tid == 0) {
    for (int d = 0; d < 8; d++) {
        O[0 * 8 + d] = A[d] / rowSum;
    }
}

if (tid == 2) {
    for (int d = 0; d < 8; d++) {
        O[1 * 8 + d] = A[d] / rowSum;
    }
}

return;
}


int main() {
    int N = 2 * 8;
    std::vector<float> q = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,2.0f,2.0f,2.0f,2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> k = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f};
    std::vector<float> s(N);

    float *d_q = nullptr;
    float *d_k = nullptr;
    float *d_s = nullptr;

    size_t size = N * sizeof(float);
    
    int tpB = 8;
    int bpG = 1;

     cudaMalloc(&d_q, size);
     cudaMalloc(&d_k, size);
     cudaMalloc(&d_s, size);

     cudaMemcpy(d_q, q.data(), size, cudaMemcpyHostToDevice);
     cudaMemcpy(d_k, k.data(), size, cudaMemcpyHostToDevice);

     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);

     cudaEventRecord(start);
     dummy<<<bpG, tpB>>>(d_q,d_k, d_s);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaMemcpy(s.data(), d_s, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < s.size(); i++) {
        std::cout << s[i] << std::endl;
    }

     float gpuTimer;

     cudaEventElapsedTime(&gpuTimer, start, stop);

     std::cout << gpuTimer << "ms." << std::endl;

     

     cudaFree(d_q); cudaFree(d_k), cudaFree(d_s);

     return 0;

}