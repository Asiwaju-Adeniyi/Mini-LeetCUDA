#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

__global__ void topk(const float* __restrict__ input, float* __restrict__ values, int* __restrict__ indices, int batch_size, int n, int k) {
    int batch_idx = blockIdx.x;

    if (batch_idx < batch_size) {
        const float* input_row = input + batch_idx * n;
        float* v_row = values + batch_idx * k;
        float* i_row = indices + batch_idx * k;

        for (int i = 0; i < k; i++) {
        v_row[i] = -INFINITY;
        i_row[i] = -1;
    }
    }




    for (int i = 0; i < n; i++) {
        float val = input_row[i];

        for (int j = 0; j < k; j++) {
            if (val > v_row[j]) {
                for (int m = k - 1; m > j; m--) {
                    v_row[m] = v_row[m-1];
                    i_row[m] = i_row[m-1];
                }

                v_row[j] = val;
                i_row[j] = i;
                break;
            }
        }
    }
}