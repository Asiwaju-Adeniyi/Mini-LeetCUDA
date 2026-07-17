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

    __device__ __forceinline__ operator>(const IndexValue& other) const {
        return value > other.value;
    }
    __device__ __forceinline__ operator<(const IndexValue& other) const {
        return value < other.value;
    }
)