#!/usr/bin/env python3
"""Generate the gemm/figures/*.png plots from a bench_all.csv sweep.

Usage (any venv with matplotlib):
    .venv/bin/python gemm/figures/make_figures.py [path/to/bench_all.csv]

Defaults to the Oct 27 2026 A10 sweep (`modal/reports/20261027T065855Z/`).
Produces, per dtype, a GFLOP/s-vs-size chart with the cuBLAS reference
overlay, plus a "blocktiling_2d as % of cuBLAS" chart.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    REPO_ROOT / "modal" / "reports" / "20261027T065855Z" / "bench_all.csv"
)
OUT_DIR = Path(__file__).resolve().parent

SIZES = [128, 256, 512, 1024, 2048, 4096]

SERIES = {
    "f32": [
        ("naive_f32", "naive (coalesced)"),
        ("shared_f32", "shared memory"),
        ("blocktiling_1d_f32", "blocktiling 1D (TM=8)"),
        ("blocktiling_2d_f32", "blocktiling 2D (8x8, float4)"),
        ("cublas_sgemm", "cuBLAS SGEMM"),
    ],
    "f16": [
        ("naive_f16", "naive (coalesced)"),
        ("shared_f16", "shared memory"),
        ("blocktiling_1d_f16", "blocktiling 1D (TM=8)"),
        ("blocktiling_2d_f16", "blocktiling 2D (8x8, float4)"),
        ("cublas_hgemm", "cuBLAS HGEMM (tensor cores)"),
    ],
}

STYLE = {
    "naive (coalesced)": dict(color="tab:gray", marker="o"),
    "shared memory": dict(color="tab:purple", marker="s"),
    "blocktiling 1D (TM=8)": dict(color="tab:orange", marker="^"),
    "blocktiling 2D (8x8, float4)": dict(color="tab:blue", marker="v"),
    "cuBLAS SGEMM": dict(color="tab:green", ls="--", marker="*", lw=2),
    "cuBLAS HGEMM (tensor cores)": dict(color="tab:red", ls="--", marker="*", lw=2),
}

FIG_KWARGS = dict(figsize=(7.5, 4.5), dpi=150)


def load_medians(csv_path: Path) -> dict[tuple[str, int], float]:
    import csv

    grouped: dict[tuple[str, int], list[float]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            key = (row["kernel"], int(row["size"]))
            grouped.setdefault(key, []).append(float(row["gflops"]))
    return {k: statistics.median(v) for k, v in grouped.items()}


def gflops_chart(medians, dtype: str, path: Path) -> None:
    fig, ax = plt.subplots(**FIG_KWARGS)
    for kernel, label in SERIES[dtype]:
        ys = [medians.get((kernel, s)) for s in SIZES]
        ax.plot(
            SIZES, ys, label=label, color=STYLE[label].get("color"),
            marker=STYLE[label].get("marker", "o"),
            ls=STYLE[label].get("ls", "-"), lw=STYLE[label].get("lw", 1.5),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("GEMM size  M = N = K")
    ax.set_ylabel("GFLOP/s (median of 3 reps)")
    dtype_name = {"f32": "fp32", "f16": "fp16 storage, fp32 scalar accumulate"}[dtype]
    ref = "SGEMM" if dtype == "f32" else "HGEMM (tensor cores)"
    ax.set_title(
        f"GEMM ladder on NVIDIA A10 (sm_86), {dtype_name}\n"
        f"vs cuBLAS {ref}"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ratio_chart(medians, path: Path) -> None:
    fig, ax = plt.subplots(**FIG_KWARGS)
    for dtype, hand, ref, color in (
        ("f32", "blocktiling_2d_f32", "cublas_sgemm", "tab:blue"),
        ("f16", "blocktiling_2d_f16", "cublas_hgemm", "tab:red"),
    ):
        ratios = [
            100 * medians[(hand, s)] / medians[(ref, s)] for s in SIZES
        ]
        ax.plot(
            SIZES, ratios, marker="o",
            label={
                "f32": "blocktiling 2D (f32) as % of cuBLAS SGEMM",
                "f16": "blocktiling 2D (f16) as % of cuBLAS HGEMM",
            }[dtype],
            color=color,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("GEMM size  M = N = K")
    ax.set_ylabel("% of cuBLAS")
    ax.set_ylim(0, 100)
    ax.set_title("Best hand-written kernel vs cuBLAS baseline (A10)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    medians = load_medians(csv_path)
    gflops_chart(medians, "f32", OUT_DIR / "f32_gflops_vs_size.png")
    gflops_chart(medians, "f16", OUT_DIR / "f16_gflops_vs_size.png")
    ratio_chart(medians, OUT_DIR / "blocktiling_2d_vs_cublas.png")
    print(f"wrote 3 figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
