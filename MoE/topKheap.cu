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

    if (left < size && heap[left] < heap[smallest]) {smallest = left;}
    if (right < size && heap[right] < heap[smallest]) {smallest = right;}
    
    if (smallest != k) {
        IndexValue temp = heap[k];
        heap[k] = heap[smallest];
        heap[smallest] = temp;

        heapify_down(heap, smallest, size);
    }

}