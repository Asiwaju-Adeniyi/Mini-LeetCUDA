# Mini-LeetCUDA

A personal deep-dive into CUDA C++ through first-principles kernel design, optimization, and performance analysis.

---

## Motivation

This repository is inspired by the idea of *learning by rebuilding*.

Projects like **LeetGPU**, **LeetCUDA** show that the fastest way to truly understand GPU programming is not by reading or copying kernels—but by **reconstructing them from scratch**, iteratively, while measuring performance and reasoning about hardware behavior.

Mini-LeetCUDA is my attempt to take that idea seriously.

Instead of passively studying CUDA kernels, this repo is focused on:

- Re-implementing core deep learning primitives from scratch  
- Progressively optimizing them from naive → production-grade  
- Understanding *why* each optimization works at the hardware level  
- Building intuition for GPU execution, memory hierarchy, and performance bottlenecks  

---

## Goals

The goal is simple but demanding:

> Take a small set of fundamental kernels and implement them at multiple levels of optimization, while developing a deep mental model of GPU execution.

### Kernels Covered

- Elementwise / vector add
- Reduction
- GEMM (matrix multiplication)
- Softmax
- LayerNorm
- Mixture-of-Experts building blocks (GEMV, top-k)
- Attention (FlashAttention-style)

---

## Approach

Each kernel is implemented in three stages:

### 1. Naive Implementation
- Straightforward mapping from math → threads
- Minimal optimization
- Focus on correctness and clarity

### 2. Optimized Implementation
- Shared memory usage
- Coalesced memory access
- Basic tiling
- Warp-level primitives where applicable

### 3. Fully Optimized Implementation
- Advanced tiling strategies
- Register blocking
- Warp specialization
- Double buffering / pipelining
- Tensor Core usage (WMMA / MMA where applicable)
- Memory layout transformations (swizzling, packing)

---

## Workflow

Each kernel follows the same development pipeline:

1. **CUDA Kernel Implementation**  
2. **PyTorch Bindings** (C++/CUDA extension)  
3. **Unit Tests** (correctness vs PyTorch reference)  
4. **Benchmarking** (throughput, latency)  
5. **Profiling** (Nsight Systems / Nsight Compute)  
6. **Documentation** (write-ups explaining design decisions)

---

## Compiling and profiling on Modal

No local CUDA toolkit is required. `modal/run_cuda.py` copies this repository
into a CUDA 12.4 image, compiles for the T4 (`sm_75`), and runs Nsight Compute.

Compile, run, and profile one standalone kernel:

```bash
modal run modal/run_cuda.py \
  --kernel gemm/naive_f32.cu \
  --name sgemm_naive
```

`ncu` reports are saved to the `minileetcuda-reports` Modal volume. Inspect a
report with:

```bash
ncu --import <report>.ncu-rep --page details
```

---

## Repository Structure

```
Mini-LeetCUDA/
├── common/              # Shared CUDA helpers (checks, timing, reductions)
├── reduction/           # Reduction kernels
├── gemm/                # Matrix multiplication kernels
├── softmax/             # Softmax + online softmax implementations
├── layernorm/           # LayerNorm implementations
├── MoE/                 # GEMV and top-k kernels
├── attention/           # Attention / FlashAttention-style kernels
├── vector addition/     # Elementwise add kernels
├── CudaForDLClassQuizzes/
├── CuteAlgebra-Cutlass/
├── trainingloop/
└── modal/               # Remote T4 compile/profile runner
```

The GEMM files use a dtype suffix:

- `*_f32.cu` — fp32 storage and math
- `*_f16.cu` — fp16 storage with fp32 accumulation

The GEMM files are standalone: they allocate inputs, run the kernel, check the
result against a CPU reference, and benchmark with CUDA events. Shared helpers
live in `common/`:

- `common/cuda_utils.cuh` — `CUDA_CHECK`, random initialization, event timing,
  GFLOP/s calculation, and sampled GEMM verification
- `common/reductions.cuh` — warp- and block-wide sum / max reductions

---

## Key Learning Areas

This project focuses heavily on understanding:

- Thread/block/warp mapping
- Memory hierarchy (global, shared, registers)
- Memory coalescing and access patterns
- Synchronization and warp-level primitives
- Arithmetic intensity and roofline thinking
- Tensor Core programming (WMMA / MMA)
- Kernel fusion and pipeline design

---

## Tooling

- CUDA C++
- PyTorch C++ Extensions
- Nsight Systems (`nsys`)
- Nsight Compute (`ncu`)

---

## Why This Exists

Most CUDA learning resources stop at explaining *what* kernels do.

This project is about understanding:

- How kernels map to hardware
- Why certain designs outperform others
- What trade-offs exist between compute, memory, and parallelism

The objective is not just to write working kernels—but to develop the ability to **reason about performance like an HPC engineer**.

---

## Progress Philosophy

- Start simple, then iterate
- Measure everything
- Never trust intuition without profiling
- Treat every kernel like a system, not just code

---

## Disclaimer

This is a learning-focused repository.  
Expect incremental progress, experiments, and evolving implementations.

---

## Future Extensions

- FlashAttention backward pass  
- FP8 kernels  
- CUTLASS-based comparisons  
- Triton equivalents  
- Multi-GPU scaling experiments  

---

## Acknowledgment

Inspired by the philosophy behind LeetGPU and the broader CUDA performance engineering community.

---

## Resources 
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main).
- [CUDA for Deep Learning](https://www.manning.com/books/cuda-for-deep-learning) by Elliot Arledge. 
- Simon Boehm's [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM). 
- [Modern GPU programming for MLsys](https://mlc.ai/modern-gpu-programming-for-mlsys/).
- [How To Write a Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html).

