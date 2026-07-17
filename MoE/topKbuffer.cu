#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

__global__ void topkBuffer(const float* __restrict__ input, float* __restrict__ values, 
    int* __restrict__ indices, int n, int k, bool* __restrict__ selected) {
        int row = blockIdx.x; 

        if (row < 1) {
            for (int i = 0; i < N; i++) {
                selected[i] = false;
            }

            for (int k = 0; k < K; k++) {
                float maxVal = -INFINITY;
                float maxIdx = -1;

                for (int i = 0; i < N; i++) {
                    if (!selected[i] && input[i] > maxVal) {
                        maxVal[i] = input[i];
                        maxIdx = i;
                    }
                }

                if (maxIdx >= 0) {
                    selected[maxIdx] = true;
                    values[k] = maxVal;
                    indiced[k] = maxIdx;
                }
            }
        
        }

        }
