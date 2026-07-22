from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_beamforming_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical local-array and distributed beamforming baselines.")
    parser.add_argument("--output", type=Path, default=Path("research/results/beamforming"))
    parser.add_argument("--study", choices=("all", "local", "distributed"), default="all")
    parser.add_argument("--quality", choices=("preview", "simulation", "fine", "reference"), default="preview")
    parser.add_argument("--floorplan-idx", type=int, default=0)
    parser.add_argument("--distributed-nodes", type=int, default=8)
    parser.add_argument("--subset-counts", type=int, nargs="+", default=(2, 4, 6, 8))
    parser.add_argument("--duration", type=float, default=2.4)
    parser.add_argument("--rir-duration", type=float, default=1.0)
    parser.add_argument("--target-interferer-snr", type=float, default=0.0)
    parser.add_argument("--sensor-noise-snr", type=float, default=18.0)
    parser.add_argument("--rt-accelerator", choices=("numba", "cuda"), default="numba")
    parser.add_argument("--rt-precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--rt-cuda-device", type=int, default=0)
    arguments = parser.parse_args()
    payload = run_beamforming_study(
        arguments.output,
        study=arguments.study,
        quality=arguments.quality,
        floorplan_idx=arguments.floorplan_idx,
        distributed_nodes=arguments.distributed_nodes,
        subset_counts=arguments.subset_counts,
        duration_s=arguments.duration,
        rir_duration_s=arguments.rir_duration,
        target_interferer_snr_db=arguments.target_interferer_snr,
        target_sensor_noise_snr_db=arguments.sensor_noise_snr,
        rt_accelerator=arguments.rt_accelerator,
        rt_precision=arguments.rt_precision,
        rt_cuda_device=arguments.rt_cuda_device,
    )
    print(arguments.output / "report.md")
    print(f"rows={len(payload['results'])}")


if __name__ == "__main__":
    main()
