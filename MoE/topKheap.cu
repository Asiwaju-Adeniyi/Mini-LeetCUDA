#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>

struct IndexValue (
    float value;
    int index; 

    __device__ __forceinline__ bool operator>(const IndexValue& other) const {
        return value > other.value;
    }
    __device__ __forceinline__ bool operator<(const IndexValue& other) const {
        return value < other.value;
    }
)


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

    heapify_down(heap, k, size);

}
}


__global__ void topK_Heap(float* __restrict__ input, int* __restrict__ indices, float* value, int K, int N) {

    extern __shared__ float heap[];

    if (threadIdx.x < K) {
        heap.value[threadIdx.x] = input[threadIdx.x];
        heap.index[threadIdx.x] = input[threadIdx.x];
    }

 if (threadIdx.x == 0) {for (int i = K/2 - 1; i >= 0; --i) {
        heapify_down(heap, i, K);
    }}

 __syncthreads();


}