#!/usr/bin/env python3
"""Generate cutlass_practice/figures PNGs for the CuTe v0 cross-GPU story.

Unlike the gemm sweep (one kernel, many sizes), this practice keeps the
problem fixed at M = N = K = 2048 and moves it across GPUs, so the chart is
a grouped bar pair per GPU rather than a size sweep.

Usage (any venv with matplotlib):
    python figures/make_figures.py

Data: every modal/reports/*/bench_all.csv on disk is scanned for
`matmul_cute_v0` and `cublas_hgemm` rows at size 2048 (medians computed per
GPU). The T4 baseline is hardcoded — it predates the runner and was measured
on Colab (numbers from the README).
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "modal" / "reports"
OUT_DIR = Path(__file__).resolve().parent

SIZE = 2048
KERNELS = ("matmul_cute_v0", "cublas_hgemm")

# (arch, display-name, measured-where) — plot order matters.
GPUS = [
    ("sm_75", "T4 (sm_75)\nColab", "colab"),
    ("sm_90", "H100 SXM\n(sm_90)", "modal"),
    ("sm_100", "B200\n(sm_100)", "modal"),
]

# Colab baseline, from the README (v0 measured twice pre-runner).
T4 = {
    "matmul_cute_v0": (2.375, 7.23),
    "cublas_hgemm": (0.406, 42.36),
}

STYLE = {
    "matmul_cute_v0": dict(color="tab:blue", label="v0: naive CuTe"),
    "cublas_hgemm": dict(color="tab:red", label="cuBLAS HGEMM"),
}

FIG_KWARGS = dict(dpi=150)


def load_medians(reports_root: Path) -> dict[str, dict[str, float]]:
    """gpu-arch -> kernel -> median GFLOP/s across all downloaded runs."""
    grouped: dict[tuple[str, str], list[float]] = {}
    csvs = sorted(reports_root.glob("*/bench_all.csv"))
    if not csvs:
        raise SystemExit(f"no bench_all.csv files under {reports_root}")
    for csv_path in csvs:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                kernel = row["kernel"]
                if kernel not in KERNELS or int(row["size"]) != SIZE:
                    continue
                if row["arch"] == "sm_75":
                    continue  # T4 predates the runner; hardcoded below
                grouped.setdefault((row["arch"], kernel), []).append(
                    float(row["gflops"])
                )
    return_dict: dict[str, dict[str, float]] = {}
    for (arch, kernel), values in grouped.items():
        entry = return_dict.setdefault(arch, {})
        entry[kernel] = statistics.median(values)
    return return_dict


def collect() -> dict[str, dict[str, float]]:
    """gpu-arch -> {kernel: (ms, TFLOP/s)} for the three target GPUs."""
    modal = load_medians(REPORTS_ROOT)
    data: dict[str, dict[str, float]] = {}
    for arch, _, where in GPUS:
        if where == "colab":
            data[arch] = {k: (ms, tf) for k, (ms, tf) in T4.items()}
        elif arch in modal:
            data[arch] = {
                kernel: (None, gflops / 1000.0)
                for kernel, gflops in modal[arch].items()
            }
        else:
            print(f"note: no bench rows for {arch} yet; skipping in figure")
    return data


def two_panel(data: dict[str, dict[str, dict[str, float]]], path: Path) -> None:
    arches = [arch for arch, _, _ in GPUS if arch in data]
    labels = [name for arch, name, _ in GPUS if arch in data]
    v0_tf = [data[arch]["matmul_cute_v0"][1] for arch in arches]
    ref_tf = [data[arch]["cublas_hgemm"][1] for arch in arches]
    pct = [100 * v / r for v, r in zip(v0_tf, ref_tf)]

    fig, (ax_speed, ax_pct) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), **FIG_KWARGS
    )

    # Panel A: absolute TFLOP/s, log scale, value-labelled bars.
    x = range(len(arches))
    width = 0.38
    bars_v0 = ax_speed.bar(
        [i - width / 2 for i in x], v0_tf, width,
        color=STYLE["matmul_cute_v0"]["color"], label="v0: naive CuTe",
    )
    bars_ref = ax_speed.bar(
        [i + width / 2 for i in x], ref_tf, width,
        color=STYLE["cublas_hgemm"]["color"], label="cuBLAS HGEMM",
    )
    ax_speed.set_yscale("log")
    ax_speed.set_ylim(1, max(ref_tf) * 3)
    ax_speed.set_xticks(list(x))
    ax_speed.set_xticklabels(labels)
    ax_speed.set_ylabel("TFLOP/s (log scale)")
    ax_speed.set_title("Absolute: v0 vs cuBLAS, 2048\u00b3 fp16")
    ax_speed.grid(True, axis="y", alpha=0.25)
    for bars, values in ((bars_v0, v0_tf), (bars_ref, ref_tf)):
        for bar, val in zip(bars, values):
            ax_speed.annotate(
                f"{val:.2f}" if val < 100 else f"{val:.0f}",
                (bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8,
            )
    ax_speed.legend(loc="upper right", fontsize=8)

    # Panel B: v0 as % of the same GPU's cuBLAS — the "naive barely scales"
    # story in one line of bars.
    bars_pct = ax_pct.bar(x, pct, 0.5, color=style_color_pct())
    ax_pct.set_ylim(0, 100)
    ax_pct.set_xticks(list(x))
    ax_pct.set_xticklabels(labels)
    ax_pct.set_ylabel("v0 as % of same-GPU cuBLAS")
    ax_pct.set_title("Relative: v0 \u00f7 cuBLAS purity penalty")
    ax_pct.grid(True, axis="y", alpha=0.25)
    for bar, val in zip(bars_pct, pct):
        ax_pct.annotate(
            f"{val:.0f}%",
            (bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        "Naive CuTe v0 vs cuBLAS HGEMM across GPUs, M = N = K = 2048",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    for name, tf0, tf1, p in zip(labels, v0_tf, ref_tf, pct):
        print(f"  {name.replace(chr(10), ' ')}: v0 {tf0:.2f} TF vs {tf1:.2f} TF ({p:.0f}%)")


def style_color_pct() -> str:
    return "tab:blue"


def main() -> None:
    data = collect()
    if len(data) < 2:
        raise SystemExit("need at least T4 + one Modal GPU; rerun when more land")
    out = OUT_DIR / "v0_vs_cublas_cross_gpu.png"
    print(f"writing {out}")
    two_panel(data, out)
    print("wrote 1 figure ->", OUT_DIR)


if __name__ == "__main__":
    main()
