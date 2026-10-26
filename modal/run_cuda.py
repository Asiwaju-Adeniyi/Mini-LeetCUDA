import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/minileetcuda"
CUDA_ARCH = "sm_75"

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


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    result.check_returncode()
    return result


@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=1800,
    volumes={"/reports": report_volume},
)
def profile_kernel(kernel_source: str, kernel_name: str):
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
        print("This kernel must be a standalone translation unit with a main().")
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


@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(
        description="Compile and profile a Mini-LeetCUDA kernel on Modal"
    )
    parser.add_argument(
        "--kernel",
        default="gemm/naive_f32.cu",
        help="Path to the .cu file, relative to the repo root",
    )
    parser.add_argument(
        "--name",
        default="sgemm_naive",
        help="Kernel name or regex to profile with ncu",
    )
    args = parser.parse_args()
    profile_kernel.remote(args.kernel, args.name)
