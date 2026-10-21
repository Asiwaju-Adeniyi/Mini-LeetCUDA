# GEMM: from naive CuTe to cuBLAS

#### All kernels are handwritten. Claude generated benchmark codes. 

Fp16 tensor-core GEMM written with CUTLASS/CuTe on a Tesla T4 (sm_75). Each kernel version is benchmarked against cuBLAS, so every optimization is measured against a fixed reference.

## Results

Problem: `M = N = K = 2048`, fp16 inputs, fp32 accumulation.

| Version | Time (ms) | TFLOP/s | % of cuBLAS | What changed |
|---|---|---|---|---|
| cuBLAS (reference) | 0.406 | 42.36 | 100% | `cublasGemmEx`, tensor-op algo |
| v0: naive CuTe | 2.375 | 7.23 | 17% | Fragments loaded straight from global memory |
| v1: shared memory | TBD | TBD | TBD | Cooperative global-to-shared copy |
| v2: vectorized copy | TBD | TBD | TBD | `TiledCopy` with 128-bit loads |
| v3: swizzled smem | TBD | TBD | TBD | Bank-conflict-free shared layout |
| v4: pipelined | TBD | TBD | TBD | Overlap next k-tile load with MMA |

## v0: naive CuTe baseline

- **Tile:** 128 x 128 x 32 per thread block, 128 threads (2 x 2 warps).
- **MMA atom:** `SM75_16x8x8_F32F16F16F32_TN` (Turing tensor core, fp16 in, fp32 accumulate).
- **Data path:** global memory directly into registers, then MMA. No shared memory, no pipelining.
- **Why it is slow:** every warp fetches its own A and B fragments from global memory, so the same tile data is loaded repeatedly and nothing hides memory latency. This is the number the later versions are meant to beat.
- **Layouts:** `A` is `M x K` row-major, `B` is stored `N x K` (K contiguous), so the kernel computes `C = A * B^T`, with `C` row-major.
- **Limitations:** no bounds handling, so M, N, K must be multiples of 128, 128, 32.

## Methodology

| | |
|---|---|
| GPU | Tesla T4 (sm_75), Google Colab |
| CUDA | 13.0 (nvcc 13.0.88) |
| CUTLASS | `<commit hash>` |
| Timing | `cudaEvent`, average over 20 iterations after 3 warmup launches |
| Correctness | Compared elementwise against a plain CUDA reference kernel (fp32 accumulation). Inputs are multiples of 0.25 so the comparison is exact: 0 mismatches, max abs error 0 |
| cuBLAS call | `cublasGemmEx`, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP` |

**Caveat:** Colab T4 clocks vary with thermals and power limits, so absolute numbers shift between runs. The table reports `<N>` runs; the figure shown is the median. (Fill in after repeating the benchmark. v0 measured 2.389 ms and 2.375 ms on two separate runs.)

## Reproduce

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git

nvcc -std=c++17 --expt-relaxed-constexpr -arch=sm_75 \
  -I cutlass/include -I cutlass/tools/util/include \
  cute_gemm.cu -o cute_gemm -lcublas && ./cute_gemm
```

Expected output format:

```
mismatches: 0, max abs error: 0
avg time: 2.375 ms, 7.23 TFLOP/s
cuBLAS: 0.406 ms, 42.36 TFLOP/s
```

## Next steps

Each step adds one row to the table above, timed with the same harness: shared memory staging, vectorized `TiledCopy`, swizzled shared layout, then software pipelining.