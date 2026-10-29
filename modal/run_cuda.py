#!/usr/bin/env python3
"""Compile, run, benchmark, and NCU-profile Mini-LeetCUDA kernels on Modal.

Examples:
  # Warm up and profile the two naive GEMMs on an A10 at 4096^3.
  modal run modal/run_cuda.py --suite naive --mode both --size 4096 --gpu a10 --open-reports

  # Benchmark every GEMM kernel at 128..4096 (3 reps, median in the plots).
  modal run modal/run_cuda.py --suite all --mode sweep --gpu a10

  # NCU-profile every GEMM kernel at 4096^3 (one timed launch each).
  modal run modal/run_cuda.py --suite all --mode ncu-sweep --size 4096 --gpu a10

  # Cutlass practice: NCU-profile the naive CuTe v0 matmul on Hopper
  # (bench + NCU + cuBLAS-at-2048 baseline in two runs).
  modal run modal/run_cuda.py --suite cutlass --mode both --gpu h100 --open-reports
  modal run modal/run_cuda.py --suite cublas --mode warmup --size 2048 --gpu h100

  # Do all of the gemm-suite steps in one container.
  modal run modal/run_cuda.py --suite all --mode all --size 4096 --gpu a10
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/minileetcuda"
REPORTS_MOUNT = "/reports"
REPORTS_SUBDIR = "gemm"

# CUTLASS headers for the CuteAlgebra-Cutlass practice kernels.  Downloaded
# from the release tarball (no git in the image) and unpacked flat so
# `-I /opt/cutlass/include` picks up cute/tensor.hpp.
CUTLASS_TAG = "v4.8.1"  # commit 2c85f49, optimal codegen on CUDA 13.3
CUTLASS_HOME = "/opt/cutlass"

# The kernels use 5 warmup launches and 50 timed launches.  In NCU we profile
# the first timed launch, so launch-skip = 1 correctness launch + 5 warmups.
BENCH_WARMUP = 5
BENCH_ITERS = 50

DEFAULT_SIZES = [128, 256, 512, 1024, 2048, 4096]

GPU_CONFIGS = {
    "a10": {"gpu": "A10", "arch": "sm_86", "label": "NVIDIA A10"},
    "a100": {"gpu": "A100-40GB", "arch": "sm_80", "label": "NVIDIA A100 40GB"},
    # "!" pins the H100: Modal may otherwise silently upgrade to an H200,
    # which would swap bandwidth/memory-size numbers under the benchmark.
    "h100": {"gpu": "H100!", "arch": "sm_90", "label": "NVIDIA H100 SXM"},
    # Blackwell (sm_100) needs CUDA >= 12.8; 13.3.1 is the toolkit CUTLASS
    # recommends for code generation on Blackwell.
    "b200": {"gpu": "B200", "arch": "sm_100", "label": "NVIDIA B200"},
}

CUDA_IMAGE_TAG = {
    "sm_75": "12.4.0",
    "sm_80": "12.4.0",
    "sm_86": "12.4.0",
    "sm_90": "12.4.0",
    "sm_100": "13.3.1",
}


def make_cuda_image(arch: str) -> modal.Image:
    tag = CUDA_IMAGE_TAG[arch]
    return (
        modal.Image.from_registry(
            f"nvidia/cuda:{tag}-devel-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])
        .run_commands(
            "apt-get update",
            "apt-get install -y --no-install-recommends ca-certificates curl",
            (
                f"curl -fsSL https://github.com/NVIDIA/cutlass/archive/refs/tags/"
                f"{CUTLASS_TAG}.tar.gz -o /tmp/cutlass.tar.gz"
                f" && mkdir -p {CUTLASS_HOME}"
                f" && tar xzf /tmp/cutlass.tar.gz -C {CUTLASS_HOME}"
                f" --strip-components=1"
                f" && rm /tmp/cutlass.tar.gz"
            ),
        )
        .add_local_dir(
            str(REPO_ROOT),
            remote_path=REMOTE_ROOT,
            ignore=[
                ".git",
                "**/__pycache__",
                "**/*.ncu-rep",
                "**/*.csv",
                "**/.DS_Store",
                ".DS_Store",
                "modal/reports",
                "gemm/figures",
            ],
        )
    )


IMAGES = {arch: make_cuda_image(arch) for arch in CUDA_IMAGE_TAG}

# ``kernel`` is the regex passed to ncu.  It may match a template
# instantiation, e.g. ``blocktiling_2d_f16<128, 128, 16, 8, 8>``.
# ``print_name`` is the label the standalone binary prints.
KERNELS = [
    {
        "id": "naive_f32",
        "source": "gemm/naive_f32.cu",
        "kernel": "sgemm_naive",
        "print_name": "sgemm_naive",
        "dtype": "f32",
        "display": "Naive (coalesced)",
    },
    {
        "id": "naive_f16",
        "source": "gemm/naive_f16.cu",
        "kernel": "hGemm",
        "print_name": "hGemm",
        "dtype": "f16",
        "display": "Naive (coalesced)",
    },
    {
        "id": "shared_f32",
        "source": "gemm/shared_f32.cu",
        "kernel": "shared_sgemm",
        "print_name": "shared_sgemm",
        "dtype": "f32",
        "display": "Shared memory",
    },
    {
        "id": "shared_f16",
        "source": "gemm/shared_f16.cu",
        "kernel": "hSharedGemm",
        "print_name": "hSharedGemm",
        "dtype": "f16",
        "display": "Shared memory",
    },
    {
        "id": "blocktiling_1d_f32",
        "source": "gemm/blocktiling_1d_f32.cu",
        "kernel": "blocktiling_1d",
        "print_name": "blocktiling_1d",
        "dtype": "f32",
        "display": "1D blocktiling",
    },
    {
        "id": "blocktiling_1d_f16",
        "source": "gemm/blocktiling_1d_f16.cu",
        "kernel": "blocktiling_1d_f16",
        "print_name": "blocktiling_1d_f16",
        "dtype": "f16",
        "display": "1D blocktiling",
    },
    {
        "id": "blocktiling_2d_f32",
        "source": "gemm/blocktiling_2d_f32.cu",
        "kernel": "blocktiling_2d",
        "print_name": "blocktiling_2d",
        "dtype": "f32",
        "display": "2D blocktiling (vectorized loads)",
    },
    {
        "id": "blocktiling_2d_f16",
        "source": "gemm/blocktiling_2d_f16.cu",
        "kernel": "blocktiling_2d_f16",
        "print_name": "blocktiling_2d_f16",
        "dtype": "f16",
        "display": "2D blocktiling",
    },
    {
        "id": "cublas_sgemm",
        "source": "gemm/cublas_ref.cu",
        "kernel": None,
        "print_name": "cublas_sgemm",
        "dtype": "f32",
        "display": "cuBLAS SGEMM",
        "profile": False,
    },
    {
        "id": "cublas_hgemm",
        "source": "gemm/cublas_ref.cu",
        "kernel": None,
        "print_name": "cublas_hgemm",
        "dtype": "f16",
        "display": "cuBLAS HGEMM",
        "profile": False,
    },
]

KERNEL_BY_ID = {k["id"]: k for k in KERNELS}
SUITES = {
    "naive": ["naive_f32", "naive_f16"],
    "cublas": ["cublas_sgemm", "cublas_hgemm"],
    "all": [k["id"] for k in KERNELS],
}

app = modal.App("minileetcuda")

report_volume = modal.Volume.from_name(
    "minileetcuda-reports",
    create_if_missing=True,
)

# ``kernel`` is the regex passed to ncu.  It may match a template
# instantiation, e.g. ``blocktiling_2d_f16<128, 128, 16, 8, 8>``.
# ``print_name`` is the label the standalone binary prints.
#
# The old gemm kernels share the runner's benchmark harness
# (``name: X ms, Y GFLOP/s``); Cutlass practice kernels keep their own
# harness, so they carry per-job regexes and timing conventions instead:
#
#  timing_re   -- (ms, flops-as-printed) capture; ``flops_factor`` converts
#                  the printed flops unit into GFLOP/s for the shared CSV.
#   mismatch_re -- plain count capture (no leading kernel name).
#   bench_size  -- the source hardcodes its problem size (no argv), so every
#                  CSV row / report file records this instead of --size.
KERNELS = [
    {
        "id": "matmul_cute_v0",
        "source": "CuteAlgebra-Cutlass/cutlass_practice/matmul_dummy.cu",
        "kernel": "matmulCute",
        "print_name": "matmulCute",
        "dtype": "f16",
        "display": "Naive CuTe v0 (SM75 atom)",
        "cutlass": True,
        "bench_size": 2048,
        "warmup": 3,
        "timed": 20,
        "timing_re": r"^avg time: ([\d.]+) ms, ([\d.]+) TFLOP/s",
        "flops_factor": 1000.0,  # printed TFLOP/s -> CSV GFLOP/s
        "mismatch_re": r"^mismatches:\s+(\d+)",
    },
    {
        "id": "naive_f32",
        "source": "gemm/naive_f32.cu",
        "kernel": "sgemm_naive",
        "print_name": "sgemm_naive",
        "dtype": "f32",
        "display": "Naive (coalesced)",
    },
    {
        "id": "naive_f16",
        "source": "gemm/naive_f16.cu",
        "kernel": "hGemm",
        "print_name": "hGemm",
        "dtype": "f16",
        "display": "Naive (coalesced)",
    },
    {
        "id": "shared_f32",
        "source": "gemm/shared_f32.cu",
        "kernel": "shared_sgemm",
        "print_name": "shared_sgemm",
        "dtype": "f32",
        "display": "Shared memory",
    },
    {
        "id": "shared_f16",
        "source": "gemm/shared_f16.cu",
        "kernel": "hSharedGemm",
        "print_name": "hSharedGemm",
        "dtype": "f16",
        "display": "Shared memory",
    },
    {
        "id": "blocktiling_1d_f32",
        "source": "gemm/blocktiling_1d_f32.cu",
        "kernel": "blocktiling_1d",
        "print_name": "blocktiling_1d",
        "dtype": "f32",
        "display": "1D blocktiling",
    },
    {
        "id": "blocktiling_1d_f16",
        "source": "gemm/blocktiling_1d_f16.cu",
        "kernel": "blocktiling_1d_f16",
        "print_name": "blocktiling_1d_f16",
        "dtype": "f16",
        "display": "1D blocktiling",
    },
    {
        "id": "blocktiling_2d_f32",
        "source": "gemm/blocktiling_2d_f32.cu",
        "kernel": "blocktiling_2d",
        "print_name": "blocktiling_2d",
        "dtype": "f32",
        "display": "2D blocktiling (vectorized loads)",
    },
    {
        "id": "blocktiling_2d_f16",
        "source": "gemm/blocktiling_2d_f16.cu",
        "kernel": "blocktiling_2d_f16",
        "print_name": "blocktiling_2d_f16",
        "dtype": "f16",
        "display": "2D blocktiling",
    },
    {
        "id": "cublas_sgemm",
        "source": "gemm/cublas_ref.cu",
        "kernel": None,
        "print_name": "cublas_sgemm",
        "dtype": "f32",
        "display": "cuBLAS SGEMM",
        "profile": False,
    },
    {
        "id": "cublas_hgemm",
        "source": "gemm/cublas_ref.cu",
        "kernel": None,
        "print_name": "cublas_hgemm",
        "dtype": "f16",
        "display": "cuBLAS HGEMM",
        "profile": False,
    },
]

KERNEL_BY_ID = {k["id"]: k for k in KERNELS}
SUITES = {
    "naive": ["naive_f32", "naive_f16"],
    "cublas": ["cublas_sgemm", "cublas_hgemm"],
    # Cutlass practice jobs run their own harness (output text, launch
    # conventions, hardcoded problem size), so they live out of "all".
    "cutlass": ["matmul_cute_v0"],
    "all": [
        "naive_f32",
        "naive_f16",
        "shared_f32",
        "shared_f16",
        "blocktiling_1d_f32",
        "blocktiling_1d_f16",
        "blocktiling_2d_f32",
        "blocktiling_2d_f16",
        "cublas_sgemm",
        "cublas_hgemm",
    ],
}

# Volume layout per suite, so gemm and cutlass artifacts don't interleave.
REPORTS_SUBDIR_BY_SUITE = {
    "naive": "gemm",
    "cublas": "gemm",
    "all": "gemm",
    "cutlass": "cutlass",
}


# ---------------------------------------------------------------------------
# Remote helpers
# ---------------------------------------------------------------------------


def _print_result(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    _print_result(result)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def _compile(job: dict, arch: str, build_dir: Path) -> Path:
    source = Path(REMOTE_ROOT) / job["source"]
    executable = build_dir / job["id"]

    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")

    cmd = [
        "nvcc",
        "-O3",
        "-lineinfo",
        "-std=c++17",
        f"-arch={arch}",
        "-Xptxas",
        "-v",
        "-I",
        REMOTE_ROOT,
        str(source),
        "-o",
        str(executable),
    ]
    if "cublas" in source.read_text(errors="ignore"):
        cmd.append("-lcublas")
    if job.get("cutlass"):
        cmd += ["-I", f"{CUTLASS_HOME}/include", "--expt-relaxed-constexpr"]

    print(f"\n=== compiling {job['source']} for {arch} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)
    _print_result(result)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed for {job['source']}")
    return executable


def _gpu_snapshot() -> dict[str, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.sm,clocks.mem,"
                "temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return {}

    first = output.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    keys = [
        "device",
        "driver_version",
        "clock_sm_mhz",
        "clock_mem_mhz",
        "temp_c",
        "power_w",
        "util_pct",
    ]
    return dict(zip(keys, parts))


def _version_of(cmd: list[str]) -> str:
    try:
        output = subprocess.check_output(
            cmd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"<unavailable: {exc}>"
    return output


TIMING_RE = re.compile(
    r"^(\w+):\s+([\d.]+)\s+ms,\s+([\d.]+)\s+GFLOP/s\s*$", re.MULTILINE
)
MISMATCH_RE = re.compile(r"^(\w+) mismatches:\s+(\d+)\s*$", re.MULTILINE)


def _parse_benchmark_output(text: str, job: dict) -> dict[str, float | int]:
    factor = job.get("flops_factor", 1.0)

    flattened = job.get("mismatch_re")
    if flattened:
        match = re.search(flattened, text, re.MULTILINE)
        mismatches = {"": int(match.group(1)) if match else -1}
    else:
        mismatches = {
            name: int(count) for name, count in MISMATCH_RE.findall(text)
        }

    timing_re = job.get("timing_re")
    if timing_re:
        # Custom harnesses (e.g. Cutlass practice kernels) print a single
        # (ms, flops) pair with no leading name — capture it directly and
        # register it under the job's print_name.
        match = re.search(timing_re, text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"timing pattern matched nothing in:\n{text}")
        ms, printed_flops = float(match.group(1)), float(match.group(2))
        timings = {job["print_name"]: (ms, printed_flops * factor)}
    else:
        timings = {
            name: (float(ms), float(gflops))
            for name, ms, gflops in TIMING_RE.findall(text)
        }
    if job["print_name"] not in timings:
        raise RuntimeError(
            f"could not find timing for {job['print_name']} in:\n{text}"
        )

    ms, gflops = timings[job["print_name"]]
    if flattened:
        mismatch_count = mismatches.get("", -1)
    else:
        mismatch_count = mismatches.get(job["print_name"], -1)
    return {
        "ms": ms,
        "gflops": gflops,
        "mismatches": mismatch_count,
    }


def _run_benchmark(
    job: dict, executable: Path, size: int, *, timeout: int = 900
) -> dict[str, float | int]:
    print(f"\n=== running {job['id']} at {size}^3 ===")
    result = subprocess.run(
        [str(executable), str(size)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    _print_result(result)
    if result.returncode != 0:
        raise RuntimeError(f"{job['id']} failed at size {size}")
    parsed = _parse_benchmark_output(result.stdout + result.stderr, job)
    if parsed["mismatches"] != 0:
        raise RuntimeError(
            f"{job['id']} failed correctness at size {size}: "
            f"{parsed['mismatches']} mismatches"
        )
    return parsed


BENCH_FIELDS = [
    "run_id",
    "timestamp_utc",
    "phase",
    "gpu",
    "arch",
    "device",
    "driver_version",
    "dtype",
    "kernel",
    "print_name",
    "size",
    "m",
    "n",
    "k",
    "rep",
    "ms",
    "gflops",
    "mismatches",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "temp_c",
    "power_w",
    "util_pct",
]


def _append_bench_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCH_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in BENCH_FIELDS})


def _bench_row(
    *,
    run_id: str,
    phase: str,
    gpu: str,
    arch: str,
    device: dict[str, str],
    job: dict,
    size: int,
    rep: int,
    parsed: dict[str, float | int],
) -> dict:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "gpu": gpu,
        "arch": arch,
        "dtype": job["dtype"],
        "kernel": job["id"],
        "print_name": job["print_name"],
        "size": size,
        "m": size,
        "n": size,
        "k": size,
        "rep": rep,
        "ms": parsed["ms"],
        "gflops": parsed["gflops"],
        "mismatches": parsed["mismatches"],
        **device,
    }


def _export_ncu(report_file: Path, raw_csv: Path, details_txt: Path) -> None:
    print(f"=== exporting {report_file.name} to CSV/text ===")
    result = subprocess.run(
        ["ncu", "--import", str(report_file), "--page", "raw", "--csv"],
        capture_output=True,
        text=True,
    )
    raw_csv.write_text(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"failed to export raw CSV for {report_file}")

    result = subprocess.run(
        ["ncu", "--import", str(report_file), "--page", "details"],
        capture_output=True,
        text=True,
    )
    details_txt.write_text(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"failed to export details for {report_file}")

    # NCU's raw page is a wide CSV whose header is the metric names, so there
    # is no literal "Metric Name" column to look for.  Just require a header
    # plus at least one data row.
    csv_text = raw_csv.read_text(errors="ignore")
    if len(csv_text.splitlines()) < 2 or "," not in csv_text:
        raise RuntimeError(f"raw CSV for {report_file} looks empty")


def _profile_job(
    job: dict,
    executable: Path,
    size: int,
    *,
    ncu_set: str,
    run_id: str,
    subdir: str,
) -> list[str]:
    report_base = Path("/tmp") / f"{job['id']}_{size}_{run_id}"
    report_file = report_base.with_suffix(".ncu-rep")
    log_file = Path("/tmp") / f"{job['id']}_{size}_{run_id}_ncu.log"
    raw_csv = Path("/tmp") / f"{job['id']}_{size}_{run_id}_raw.csv"
    details_txt = Path("/tmp") / f"{job['id']}_{size}_{run_id}_details.txt"

    for path in (report_file, log_file, raw_csv, details_txt):
        path.unlink(missing_ok=True)

    # Correctness launch + this job's warmup launches; anything else the
    # binary launches (e.g. the reference kernel) must not match the NCU
    # kernel regex, or the skip count drifts.
    launch_skip = 1 + job.get("warmup", BENCH_WARMUP)
    cmd = [
        "ncu",
        "--set",
        ncu_set,
        "--clock-control",
        "none",
        "--kernel-name",
        f"regex:{job['kernel']}",
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        "1",
        "--target-processes",
        "all",
        "--force-overwrite",
        "-o",
        str(report_base),
        str(executable),
        str(size),
    ]

    print(f"\n=== ncu {job['id']} at {size}^3 (skip {launch_skip}) ===")
    print(" ".join(cmd))
    with log_file.open("w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=1800
        )

    log_text = log_file.read_text(errors="ignore")
    print(log_text[-4000:])

    if result.returncode != 0:
        raise RuntimeError(f"ncu failed for {job['id']} at size {size}")
    if not report_file.exists():
        raise RuntimeError(
            f"ncu did not produce {report_file}; see {log_file}"
        )
    if "No kernels were profiled" in log_text:
        raise RuntimeError(
            f"ncu matched no kernel for {job['id']} "
            f"(regex {job['kernel']!r}); see {log_file}"
        )

    destination = Path(REPORTS_MOUNT) / subdir
    destination.mkdir(parents=True, exist_ok=True)

    # Persist the raw report first: exports below are best-effort and must not
    # cost us the (expensive) profile if NCU's CSV formatter misbehaves.
    artifacts = []
    for path in (report_file, log_file):
        target = destination / path.name
        shutil.copy2(path, target)
        artifacts.append(target.name)
        print(f"Saved {target}")
    report_volume.commit()

    try:
        _export_ncu(report_file, raw_csv, details_txt)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"WARNING: NCU export failed: {exc}", file=sys.stderr)
    else:
        for path in (raw_csv, details_txt):
            target = destination / path.name
            shutil.copy2(path, target)
            artifacts.append(target.name)
            print(f"Saved {target}")
        report_volume.commit()

    return artifacts


def _write_metadata(
    *,
    run_id: str,
    gpu: str,
    arch: str,
    suite: str,
    jobs: list[dict],
    sizes: list[int],
    reps: int,
    ncu_set: str,
    device: dict[str, str],
    subdir: str,
) -> None:
    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "arch": arch,
        "suite": suite,
        "jobs": [job["id"] for job in jobs],
        "sizes": sizes,
        "reps": reps,
        "warmup_launches": BENCH_WARMUP,
        "timed_launches": BENCH_ITERS,
        "ncu_set": ncu_set,
        "device": device,
        "cuda": _version_of(["nvcc", "--version"]),
        "ncu": _version_of(["ncu", "--version"]),
    }
    destination = Path(REPORTS_MOUNT) / subdir
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"metadata_{run_id}.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    report_volume.commit()
    print(f"Saved {path}")


def _impl(config: dict) -> dict:
    root = Path(REMOTE_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    build_dir = Path("/tmp/minileetcuda_build")
    build_dir.mkdir(parents=True, exist_ok=True)

    jobs = [KERNEL_BY_ID[job_id] for job_id in config["jobs"]]
    gpu_config = GPU_CONFIGS[config["gpu"]]
    arch = gpu_config["arch"]
    subdir = config.get("subdir", REPORTS_SUBDIR)

    print(f"=== {gpu_config['gpu']} / {arch} / run {config['run_id']} ===")

    executables = {
        job["id"]: _compile(job, arch, build_dir) for job in jobs
    }

    device = _gpu_snapshot()
    print(f"GPU snapshot: {device}")

    _write_metadata(
        run_id=config["run_id"],
        gpu=gpu_config["gpu"],
        arch=arch,
        suite=config["suite"],
        jobs=jobs,
        sizes=config["sizes"],
        reps=config["reps"],
        ncu_set=config["ncu_set"],
        device=device,
        subdir=subdir,
    )

    bench_csv = Path(REPORTS_MOUNT) / subdir / "bench_all.csv"

    def record(job: dict, size: int, rep: int, phase: str) -> None:
        # Practice kernels hardcode their problem size in main(); trust the
        # job's bench_size so CSV rows/report names match reality.
        size = job.get("bench_size", size)
        parsed = _run_benchmark(job, executables[job["id"]], size)
        snapshot = _gpu_snapshot()
        row = _bench_row(
            run_id=config["run_id"],
            phase=phase,
            gpu=gpu_config["gpu"],
            arch=arch,
            device=snapshot or device,
            job=job,
            size=size,
            rep=rep,
            parsed=parsed,
        )
        _append_bench_row(bench_csv, row)
        report_volume.commit()
        print(
            f"recorded {job['id']} size={size} rep={rep}: "
            f"{parsed['ms']:.3f} ms, {parsed['gflops']:.2f} GFLOP/s"
        )

    artifacts: list[str] = []

    # Warm up first, then profile.  This mirrors the requested workflow and
    # keeps the profiled launch in a warm-clock, warm-cache state.
    if config["mode"] in ("warmup", "both", "all"):
        for job in jobs:
            record(job, config["size"], 0, "warmup")

    if config["mode"] in ("profile", "ncu-sweep"):
        # These modes do not run the normal benchmark first, so burn the GPU
        # briefly to ramp clocks before profiling.
        if config["gpu_warmup"]:
            record(jobs[0], config["size"], 0, "gpu_warmup")

    if config["mode"] in ("profile", "both", "ncu-sweep", "all"):
        for job in jobs:
            if not job.get("profile", True):
                print(f"skipping NCU for {job['id']} (library kernel)")
                continue
            artifacts.extend(
                _profile_job(
                    job,
                    executables[job["id"]],
                    job.get("bench_size", config["size"]),
                    ncu_set=config["ncu_set"],
                    run_id=config["run_id"],
                    subdir=subdir,
                )
            )

    if config["mode"] in ("sweep", "all"):
        for rep in range(config["reps"]):
            for size in config["sizes"]:
                for job in jobs:
                    record(job, size, rep, "sweep")

    return {
        "run_id": config["run_id"],
        "artifacts": artifacts,
        "bench_csv": bench_csv.name,
        "subdir": subdir,
        "gpu": gpu_config["gpu"],
        "arch": arch,
    }


@app.function(
    image=IMAGES["sm_86"],
    gpu="A10",
    timeout=3600,
    startup_timeout=900,
    volumes={REPORTS_MOUNT: report_volume},
)
def profile_a10(config: dict) -> dict:
    return _impl(config)


@app.function(
    image=IMAGES["sm_80"],
    gpu="A100-40GB",
    timeout=3600,
    startup_timeout=900,
    volumes={REPORTS_MOUNT: report_volume},
)
def profile_a100(config: dict) -> dict:
    return _impl(config)


@app.function(
    image=IMAGES["sm_90"],
    gpu="H100!",
    timeout=3600,
    startup_timeout=900,
    volumes={REPORTS_MOUNT: report_volume},
)
def profile_h100(config: dict) -> dict:
    return _impl(config)


@app.function(
    image=IMAGES["sm_100"],
    gpu="B200",
    timeout=3600,
    startup_timeout=900,
    volumes={REPORTS_MOUNT: report_volume},
)
def profile_b200(config: dict) -> dict:
    return _impl(config)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


def _download_file(remote_name: str, local_dir: Path, subdir: str) -> Path:
    local_path = local_dir / remote_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_path = f"{subdir}/{remote_name}"

    print(f"downloading {remote_path} -> {local_path}")
    try:
        with local_path.open("wb") as f:
            for chunk in report_volume.read_file(remote_path):
                f.write(chunk)
    except Exception as exc:
        print(f"SDK download failed ({exc}); trying `modal volume get`")
        modal_cli = shutil.which("modal")
        if modal_cli is None:
            modal_cli = str(Path(sys.executable).with_name("modal"))
        subprocess.run(
            [
                modal_cli,
                "volume",
                "get",
                "minileetcuda-reports",
                remote_path,
                str(local_path),
            ],
            check=True,
        )
    return local_path


@app.local_entrypoint()
def main(
    suite: Literal["naive", "cublas", "cutlass", "all"] = "naive",
    gpu: Literal["a10", "a100", "h100", "b200"] = "a10",
    mode: Literal["warmup", "profile", "both", "sweep", "ncu-sweep", "all"] = "both",
    size: int = 4096,
    sizes: str = ",".join(str(s) for s in DEFAULT_SIZES),
    reps: int = 3,
    ncu_set: Literal["basic", "detailed", "full"] = "full",
    download_dir: str = str(REPO_ROOT / "modal" / "reports"),
    open_reports: bool = False,
    gpu_warmup: bool = True,
) -> None:
    size_list = [int(s) for s in sizes.split(",") if s.strip()]
    if not size_list:
        raise ValueError("--sizes must contain at least one integer")
    if size <= 0 or any(s <= 0 for s in size_list) or reps <= 0:
        raise ValueError("sizes and reps must be positive")
    if size not in size_list and mode in ("sweep", "all"):
        size_list.append(size)
        size_list.sort()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {
        "run_id": run_id,
        "jobs": SUITES[suite],
        "suite": suite,
        "subdir": REPORTS_SUBDIR_BY_SUITE[suite],
        "gpu": gpu,
        "mode": mode,
        "size": size,
        "sizes": size_list,
        "reps": reps,
        "ncu_set": ncu_set,
        "gpu_warmup": gpu_warmup,
    }

    remote_by_gpu = {
        "a10": profile_a10,
        "a100": profile_a100,
        "h100": profile_h100,
        "b200": profile_b200,
    }
    remote = remote_by_gpu[gpu]
    result = remote.remote(config)
    print("\n=== remote result ===")
    print(json.dumps(result, indent=2))

    local_dir = Path(download_dir) / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    subdir = result.get("subdir", REPORTS_SUBDIR)

    # The benchmark CSV is shared across runs; always fetch the latest copy.
    downloads = list(result.get("artifacts", []))
    if result.get("bench_csv"):
        downloads.append(result["bench_csv"])

    local_reports = []
    for name in downloads:
        path = _download_file(name, local_dir, subdir)
        if path.suffix == ".ncu-rep":
            local_reports.append(path)

    print(f"\nDownloaded {len(downloads)} file(s) to {local_dir}")

    if open_reports:
        for path in local_reports:
            print(f"opening {path} in Nsight Compute")
            subprocess.run(
                ["open", "-a", "NVIDIA Nsight Compute", str(path)],
                check=False,
            )
