#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>

#define HALF2(value) (reinterpret_cast<half2 *> (&value)[0])

