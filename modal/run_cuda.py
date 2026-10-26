import modal
import subprocess
import shutil

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
    .add_local_file(
        "vector addition/elemwise_f32.cu",
        remote_path="/root/elemwise_f32.cu",
    )
)

@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=600,
    volumes={"/reports": report_volume},
)
def profile_kernel():
    source = "/root/elemwise_f32.cu"
    output = "/tmp/elemwise_f32"
    report = "/tmp/elemwise_f32_report"

    subprocess.run(
        [
            "nvcc", "-O3", "-lineinfo", "-arch=sm_75",
            source, "-o", output,
        ],
        check=True,
    )

    subprocess.run(
        [
            "ncu",
            "--set", "basic",
            "--clock-control", "none",
            "--kernel-name", "regex:naiveVecAdd",
            "-o", report,
            output,
        ],
        check=True,
    )

    shutil.copy2(
        report + ".ncu-rep",
        "/reports/elemwise_f32_report.ncu-rep",
    )
    report_volume.commit()

    subprocess.run(
        [
            "ncu",
            "--import", report + ".ncu-rep",
            "--page", "details",
        ],
        check=True,
    )

@app.local_entrypoint()
def main():
    profile_kernel.remote()
