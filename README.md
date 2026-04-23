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

- Reduction  
- GEMM (matrix multiplication)  
- Softmax  
- LayerNorm  
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

## Repository Structure

```
mini-leetcuda/
├── reduction/     # Reduction kernels (naive → optimized → fully optimized)
├── gemm/          # Matrix multiplication kernels
├── softmax/       # Softmax + online softmax implementations
├── layernorm/     # LayerNorm and RMSNorm
├── attention/     # Attention / FlashAttention-style kernels
│
├── benchmarks/    # Performance benchmarks
├── profiling/     # Nsight Systems / Nsight Compute outputs
├── bindings/      # PyTorch C++/CUDA extensions
├── tests/         # Correctness tests vs PyTorch
└── docs/          # Explanations and kernel breakdowns
```


Each kernel folder contains:
- `naive.cu`
- `optimized.cu`
- `fully_optimized.cu`
- `README.md` (kernel-specific explanation)

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
