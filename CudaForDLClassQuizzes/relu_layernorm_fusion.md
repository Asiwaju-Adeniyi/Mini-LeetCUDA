# ReLU + LayerNorm Fused Kernel Structure

## LOAD PHASE
- Each thread loads strided elements from HBM → shared memory
```cpp
for (i = threadIdx.x; i < C; i += blockDim.x)
    shared[i] = x[i];
```
- `__syncthreads()`

---

## COMPUTE PHASE — Statistics

- Each thread accumulates partial `sum` and `sumSQ` from shared → registers
```cpp
fused.sum   += shared[j];
fused.sumSQ += shared[j] * shared[j];
```
- `blockReduc` on `fused.sum`   → register `(m)`
- `blockReduc` on `fused.sumSQ` → register `(var)`
```cpp
m   = fused.sum / C;                      // [register]
var = (fused.sumSQ / C) - (m * m);       // [register]
r   = rsqrtf(var + eps);                  // [register]
```

---

## COMPUTE PHASE — Normalize + Affine + ReLU (fused, no extra pass)

```cpp
for (k = threadIdx.x; k < C; k += blockDim.x) {
    norm   = (shared[k] - m) * r;         // [register]
    norm   = fmaxf(norm, 0.0f);           // [register — ReLU, free]
    scaled = gamma[k] * norm + beta[k];   // [register]
    y[k]   = scaled;                      // [register → HBM, once]
}
```

---

## WRITE PHASE

```cpp
mean[idx] = m;     // [register → HBM]
rstd[idx] = r;     // [register → HBM]
```

---

## Key Insight

ReLU adds exactly one instruction — `fmaxf(norm, 0.0f)` — between normalization 
and the affine transform. It costs nothing in terms of memory traffic because `norm` 
never leaves the register.

**Global memory touches for the entire kernel:**
- 1 read of the input row
- 1 read of gamma and beta
- 1 write of the output row
- 2 writes for mean and rstd

Compare to separate LayerNorm + ReLU kernels which would read and write the entire 
output row twice — once to store LayerNorm output, once to load it back for ReLU.
