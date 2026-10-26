#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

struct IndexValue {
    float value;
    int index; 

    __device__ __forceinline__ bool operator>(const IndexValue& other) const {
        return value > other.value;
    }
    __device__ __forceinline__ bool operator<(const IndexValue& other) const {
        return value < other.value;
    }
}


__device__ __forceinline__ void heapify_down(IndexValue* heap, int k, int size) {
int smallest = k;
int left = 2 * k + 1;
int right = 2 * k + 2;

if (left < size && heap[left] < heap[smallest]) smallest = left;
if (right < size && heap[right] < heap[smallest]) smallest = right;

if (smallest != k) {
    IndexValue accum = heap[k];
    heap[k] = heap[smallest];
    heap[smallest] = accum;

    heapify_down(heap, smallest, size);

}
}


__global__ void topK_Heap(float* __restrict__ input, int* __restrict__ indices, float* value, int K, int N) {

    extern __shared__ IndexValue heap[];

    if (threadIdx.x < K) {
        heap[threadIdx.x].value = input[threadIdx.x];
        heap[threadIdx.x].index = threadIdx.x;
    }

 if (threadIdx.x == 0) {for (int i = K/2 - 1; i >= 0; i--) {
        heapify_down(heap, i, K);
    }}

 __syncthreads();

 if (threadIdx.x == 0) {
    for (int i = K; i < N; i++) {
        if (input[i] > heap[0].value) {
            heap[0].value = input[i];
            heap[0].index = i;

            heapify_down(heap, 0, K);
        }
    }
 }
 __syncthreads();


 if (threadIdx.x < K) {
    values[threadIdx.x] = heap[threadIdx.x].value;
    indices[threadIdx.x] = heap[threadIdx.x].index;
 }
}