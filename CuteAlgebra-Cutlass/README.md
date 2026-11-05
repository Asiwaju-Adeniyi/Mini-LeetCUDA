# CuTe Algebra & CUTLASS Practice

Two tracks live here:

| Track | What it is |
|---|---|
| [CUTE Layout Representation and Algebra — Study Notes](CUTE_ALGEBRA_NOTES.md) | Page-by-page notes on Cecka's CUTE preprint and Colfax's categorical-foundations material, with worked examples. |
| [CUTLASS/CuTe Practice: Ampere → Hopper → Blackwell](cutlass_practice/README.md) | Handwritten tensor-core GEMM kernels, benchmarks against cuBLAS, NCU profiles, and the architecture-by-architecture roadmap. |

## Layout

```text
CuteAlgebra-Cutlass/
├── README.md                  # this index
├── CUTE_ALGEBRA_NOTES.md      # CuTe layout/algebra study notes
└── cutlass_practice/
    ├── README.md              # benchmarks, kernel notes, roadmap
    ├── Ampere/                # naive CuTe v0 + explicit SM80 TiledMMA
    ├── Hopper/                # WGMMA / TMA / pipelining track
    ├── Blackwell/             # UMMA / tensor-memory track
    ├── figures/               # generated benchmark plots
    ├── tiledCopy.cu           # WIP TiledCopy exercise
    └── vedAdd.cu              # WIP CuTe vector-add exercise
```

## Current status

- **Ampere**: `01_naive_cute.cu` is benchmarked on T4/H100/B200; `02_tiled_mma_sm80.cu` on H100/B200.
- **Hopper**: `01_wgmma_sm90.cu` is a scaffold for the Colfax WGMMA tutorial.
- **Blackwell**: `01_mma_sm100.cu` is a scaffold for the Colfax tensor-memory series.

All benchmark tables, NCU findings, figures, and reproduction commands are in
[`cutlass_practice/README.md`](cutlass_practice/README.md).
