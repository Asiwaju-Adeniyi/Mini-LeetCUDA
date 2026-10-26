import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/minileetcuda"
CUDA_ARCH = "sm_75"  # T4

# These are not standalone translation units (yet).
SKIP_DIRS = {"CuteAlgebra-Cutlass"}
SKIP_FILES = {
    "attention/flashv1.cu",  # CuTe fragment, not a standalone file
    "trainingloop/tl.cu",    # work in progress
}

app = modal.App("minileetcuda")

report_volume = modal.Volume.from_name(
    "minileetcuda-reports",
    create_if_missing=True,
)

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.ncu-rep",
            "**/.DS_Store",
            ".DS_Store",
        ],
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, *, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check:
        result.check_returncode()
    return result


def _should_compile(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    return str(rel) not in SKIP_FILES


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=1800,
)
def compile_all():
    """Compile every standalone .cu file with `nvcc -c` (no link)."""
    root = Path(REMOTE_ROOT)
    sources = [
        path for path in sorted(root.rglob("*.cu"))
        if _should_compile(path, root)
    ]

    failed = []
    for source in sources:
        rel = source.relative_to(root)
        obj = Path("/tmp") / (str(rel).replace("/", "_") + ".o")
        print(f"\n=== compiling {rel} ===")
        result = subprocess.run(
            [
                "nvcc",
                "-O3",
                "-lineinfo",
                f"-arch={CUDA_ARCH}",
                "-I",
                str(root),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("OK")
            if result.stderr.strip():
                print(result.stderr)
        else:
            print("FAIL")
            if result.stdout.strip():
                print(result.stdout)
            if result.stderr.strip():
                print(result.stderr)
            failed.append(str(rel))

    print("\n=== compile summary ===")
    print(f"{len(sources) - len(failed)}/{len(sources)} files compiled")
    for rel in failed:
        print(f"  FAIL {rel}")

    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed to compile")


@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=1800,
    volumes={"/reports": report_volume},
)
def profile_kernel(kernel_source: str, kernel_name: str):
    """Compile one standalone kernel and profile it with Nsight Compute."""
    root = Path(REMOTE_ROOT)
    source = root / kernel_source
    executable = Path("/tmp") / (source.stem + "_bin")
    report = Path("/tmp") / (source.stem + "_report")

    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")

    print(f"=== compiling {kernel_source} ===")
    compile_result = subprocess.run(
        [
            "nvcc",
            "-O3",
            "-lineinfo",
            f"-arch={CUDA_ARCH}",
            "-I",
            str(root),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    if compile_result.stdout:
        print(compile_result.stdout)
    if compile_result.stderr:
        print(compile_result.stderr, file=sys.stderr)
    if compile_result.returncode != 0:
        print(
            "\nHint: if the error is 'undefined reference to main', this "
            "kernel has no standalone main() yet. Use --compile-all to check "
            "syntax without linking, or add a benchmark main to the file."
        )
        compile_result.check_returncode()

    print(f"\n=== profiling {kernel_name} ===")
    _run(
        [
            "ncu",
            "--set",
            "basic",
            "--clock-control",
            "none",
            "--kernel-name",
            f"regex:{kernel_name}",
            "-o",
            str(report),
            str(executable),
        ]
    )

    report_file = report.with_suffix(".ncu-rep")
    if report_file.exists():
        destination = Path("/reports") / f"{source.stem}_{kernel_name}.ncu-rep"
        shutil.copy2(report_file, destination)
        report_volume.commit()
        print(f"Saved report to {destination}")
    else:
        print("No report file was produced by ncu")

    _run(["ncu", "--import", str(report_file), "--page", "details"])


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(description="Compile/profile Mini-LeetCUDA kernels on Modal")
    parser.add_argument(
        "--kernel",
        default="gemm/naive_f32.cu",
        help="Path to the .cu file, relative to the repo root",
    )
    parser.add_argument(
        "--name",
        default="sgemm_naive",
        help="Kernel name (or regex) to profile with ncu",
    )
    parser.add_argument(
        "--compile-all",
        action="store_true",
        help="Compile every standalone kernel with nvcc -c and report failures",
    )
    args = parser.parse_args()

    if args.compile_all:
        compile_all.remote()
    else:
        profile_kernel.remote(args.kernel, args.name)
