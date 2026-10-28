# GEMM — building a matmul from naive to (almost) cuBLAS

Nine standalone kernels form the classic optimization ladder, each one a
standalone binary that allocates its inputs, verifies against a CPU reference,
and benchmarks with CUDA events (5 warmup launches + 50 timed launches,
`GFLOP/s = 2·M·N·K / time`).

| File | Idea |
|---|---|
| `naive_f32.cu` / `naive_f16.cu` | One thread per output element, no shared memory. A is broadcast, B is fetched through strided, mostly uncoalesced loads. |
| `shared_f32.cu` / `shared_f16.cu` | One thread per output element, but both matrix stages (32×32×32 tile, 1024 threads) staged in shared memory; MACs run against shared. |
| `blocktiling_1d_f32.cu` / `_f16.cu` | 64×64×16 block tile (BM=64, BN=64, BK=16); each thread owns a register tile of TM=8 rows so every A-value loaded to shared is reused TM times. 512 threads/block. |
| `blocktiling_2d_f32.cu` / `_f16.cu` | 128×128×16 block tile; each thread owns an 8×8 register tile (TM=8, TN=8, 256 threads). Global loads staged into shared with `float4` vectorized copies. |
| `cublas_ref.cu` | cuBLAS `sgemm` / `hgemm` baselines so the ladder has a target. |

`*_f16.cu` variants keep **fp16 storage with fp32 accumulation**, computed with
scalar `FMA` — no tensor cores yet. That detail is the single biggest number in
this write-up.

---

## Running and profiling

Standalone (needs `common/cuda_utils.cuh`; sizes must be multiples of the tile
sizes, default 1024):

```bash
nvcc -O3 -std=c++17 -arch=native -I.. blocktiling_2d_f32.cu -o b2d
./b2d 4096          # 4096 x 4096 x 4096
./b2d 4096 2048 512 # M=4096, N=2048, K=512
```

No local CUDA toolkit required — `modal/run_cuda.py` copies the repo into a
CUDA 12.4 container, compiles, benchmarks, and NCU-profiles. See the root
README for the full flag list; the one-shot command that produced everything
in this document:

```bash
modal run modal/run_cuda.py --suite all --mode all --size 4096 --gpu a10 --open-reports
```

Median-of-3 results land in `modal/reports/<run_id>/bench_all.csv` (all sizes
128…4096, 3 reps). `-mode ncu-sweep` additionally writes one NCU report per
kernel per size, exported as both `.ncu-rep` (GUI) and `_details.txt` /
`_raw.csv` (text).

### Profiling conventions

- GPU: NVIDIA **A10** (`sm_86`, 72 SMs, ~31 TFLOP/s fp32), driver 610.57.04.
- NCU profiles the **first timed launch**: `--launch-skip 6` (correctness
  launch + 5 warmups), `--launch-count 1`, `--set full`, `--clock-control none`
  — so every report carries the "collected without fixed GPU frequencies"
  warning, and clock numbers (matched against `bench_all.csv` snapshots)
  should be taken as the stable part, not absolute guarantees.
- Reports live in `modal/reports/20261027T065855Z/` (NCU sweep + benchmark
  sweep, the source of all numbers below), plus 1024³-only
  `20261027T065505Z/` and naive-only `20261027T065012Z/` runs.

---

## Benchmark results

Median GFLOP/s over 3 reps (A10, fp32 vs fp16-storage columns side by side;
cuBLAS rows are the reference):

**fp32 kernels**

| size | naive | shared | blocktiling 1D | blocktiling 2D | cuBLAS SGEMM |
|---|---|---|---|---|---|
| 128 | 719 | 441 | 255 | 121 | 684 |
| 256 | 1380 | 1914 | 1095 | 609 | 3948 |
| 512 | 1730 | 2169 | 4514 | 2785 | 9818 |
| 1024 | 1556 | 2291 | 5761 | 8594 | 12723 |
| 2048 | 1238 | 2020 | 4931 | **11789** | 16384 |
| 4096 | 1270 | 1955 | 4815 | **11061** | 12555 |

**fp16 storage, fp32 scalar accumulate (no tensor cores)**

| size | naive | shared | blocktiling 1D | blocktiling 2D | cuBLAS HGEMM (tensor cores) |
|---|---|---|---|---|---|
| 128 | 454 | 437 | 234 | 138 | 493 |
| 256 | 1137 | 1894 | 998 | 634 | 3874 |
| 512 | 1615 | 2142 | 4113 | 2739 | 37556 |
| 1024 | 1639 | 2294 | 5628 | 10309 | 76151 |
| 2048 | 1298 | 1951 | 5355 | **10841** | 83894 |
| 4096 | 1238 | 1880 | 5153 | **10549** | 59370 |

Jump straight to the headline: **the 2D-blocktiling f32 kernel reaches ~72% of
cuBLAS SGEMM at 2048³ and ~88% at 4096³** — without tensor cores, without
async copies, just tiling + register blocking + vectorized loads. The fp16
kernels tell the other half of the story: they are statistically the *same
speed as their fp32 twins* (~11 vs ~11 TFLOP/s) while cuBLAS HGEMM rides tensor
cores to ~84 TFLOP/s. Half-precision storage costs nothing scalar-wise and
buys nothing — until `mma` shows up.

---

## What NCU says at 4096³ (Speed-of-Light summary)

| kernel | duration | SM % | DRAM % | L1/TEX % | achieved occ. | regs/thread |
|---|---|---|---|---|---|---|
| naive_f32 | 99.8 ms | 98.6 | 43.7 | 98.6 | 99.7 % | 40 |
| naive_f16 | 104.7 ms | 98.0 | 27.3 | 98.5 | 99.5 % | 26 |
| shared_f32 | 66.9 ms | 80.5 | 27.1 | 81.2 | 66.4 % | 38 |
| shared_f16 | 69.1 ms | 78.6 | 15.6 | 79.0 | 66.7 % | 36 |
| blocktiling_1d_f32 | 24.3 ms | 73.9 | 31.4 | 74.5 | 66.2 % | 58 |
| blocktiling_1d_f16 | 21.7 ms | 88.6 | 20.1 | 89.0 | 98.3 % | 40 |
| blocktiling_2d_f32 | 10.25 ms | 58.1 | 20.6 | 82.7 | 32.6 % | 122 |
| blocktiling_2d_f16 | 10.8 ms | 68.4 | 11.3 | 44.7 | 32.4 % | 128 |

Readings worth internalizing, one per rung:

1. **naive (99.8 ms)** — "high utilization" is not "efficient". SM and L1/TEX
   throughput both sit at ~98%, occupancy ~100%, and yet the kernel delivers
   ~4% of the device's fp32 peak. Every warp is saturated *issuing redundant
   work*: one thread per output element means each warp pulls 32 different
   rows/cols of A and B, so B's stride-1-column loads replay the same sectors
   over and over (DRAM traffic at 43.7% is almost all duplication). The
   pipelines are busy; the FLOPs are not.

2. **shared (67 ms)** — staging A and B tiles once per 32-k-tile in shared
   memory cuts duration by a third (99.8 → 66.9 ms) and L1/TEX utilization
   from 98.6% to 81.2%, at the cost of occupancy (66%: 8KB of shared +
   1024-thread blocks). The MAC loop
   is unchanged — every thread still re-loads sA/sB per dot product — so the
   ceiling is instruction throughput, and DRAM is quiet (27%) rather than
   done.

3. **blocktiling 1D (24 ms)** — giving each thread TM=8 register-resident rows
   halves the number of shared reads per useful FLOP again and shifts the
   balance: SM busy 74%, DRAM 31%. This is the first kernel whose bottleneck
   is the *memory system as a whole* rather than redundant L1 replay.

4. **blocktiling 2D (10.25 ms)** — 16× fewer output tiles, `float4` staging,
   8×8 register tiles per thread, ~10× faster than naive and ~6.5× faster than
   the shared-memory version. The trade shows up
   as occupancy collapsing to 32.6% — 128 regs/thread means only ~2 blocks of
   256 threads fit per SM's 64K-register file, and the SM spends part of its
   issue slots stalled (SM busy 58%). The f32 version runs aground on shared
   memory bandwidth (L1/TEX 82.7%); the f16 twin, with lighter shared traffic
   (44.7%), starts leaning on the compute pipes (68.4%). Both readings point
   the same way: the scalar FMA path is the thing to replace next.

5. **cuBLAS** — SGEMM peaks at 16.4 TFLOP/s (2048³), i.e. ~53% of A10's fp32
   peak; the hand-written 2D tiler reaches 72–88% of that without ever
   touching tensor cores. HGEMM at 84 TFLOP/s is a different instruction-set
   conversation (`mma.m16n8k16` with fp32 accumulate) and is the next ladder
   entirely.

One caveat NCU itself prints on every report: profiling ran with
`--clock-control none` so the SM frequency reflects whatever clocks the
benchmark had ramped to (`nvidia-smi` snapshots: 1095–1290 MHz across the
run). Cross-kernel comparisons within a run are sound; absolute GFLOP/s can
wobble a few percent with the state of the die that day.

---

## Next rungs on the ladder

- **`cp.async` double-buffering** — overlap the global→shared copy of tile
  k+1 with the MAC loop of tile k (Ampere's async copy instructions; the
  explicit spine of every later TMA/wgmma pipeline).
- **Occupancy vs ILP tuning on 2D** — 128 regs/thread is the limiter at 32%
  occupancy; smaller TM/TN or `__launch_bounds__` pressure trades reusability
  for more blocks in flight.
- **Tensor cores** — replace the 8×8 scalar FMA tile with `ldmatrix` +
  `mma.sync.m16n8k16` (fp16 in, fp32 accumulate) and chase the cuBLAS HGEMM
  band; note the 8–12× gap documented above.
- Beyond Ampere (kept short): TMA + warpgroup `wgmma.mma_async` on Hopper,
  then TMEM/tcgen05 on Blackwell — same producer/consumer ideas, one
  hardware generation each.
